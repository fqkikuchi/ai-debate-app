from datetime import datetime
import io
import json
import os
import random
import re
import tempfile
import time
import uuid
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
import streamlit as st

# --------------------------------------------------
# 1. ページ初期設定 & 定数設定
# --------------------------------------------------
st.set_page_config(
    page_title="汎用AIマルチエージェント討論 (Gemini 3.6 Flash省トークン版)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 汎用AIマルチエージェント討論システム")
st.caption(
    "💡提案役 ➔ ⚡批判役 ➔ ↩️反論役(選択可) ➔ 🏆審判役 | Gemini 3.6 Flash専用・省トークン最適化版"
)

HISTORY_FILE = "discussion_history.json"
MAX_HISTORY_COUNT = 10
MODEL_NAME = "gemini-3.6-flash"  # Gemini 3.6 Flashに完全固定

# --------------------------------------------------
# 2. ローカルJSONファイル操作関数 (アトミック書き込み)
# --------------------------------------------------
def load_history_from_file() -> dict:
    """ローカルJSONファイルから過去ログを読み込む"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"履歴読み込みエラー: {e}")
            return {}
    return {}

def save_history_to_file(history_data: dict):
    """ローカルJSONファイルへ過去ログを書き出す（アトミック書き込み）"""
    try:
        sorted_keys = sorted(
            history_data.keys(),
            key=lambda k: history_data[k].get("timestamp", ""),
            reverse=True
        )
        trimmed_data = {k: history_data[k] for k in sorted_keys[:MAX_HISTORY_COUNT]}
        st.session_state.history_store = trimmed_data

        dir_name = os.path.dirname(HISTORY_FILE) or "."
        with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name, encoding="utf-8") as tf:
            json.dump(trimmed_data, tf, ensure_ascii=False, indent=2)
            temp_name = tf.name

        os.replace(temp_name, HISTORY_FILE)
    except Exception as e:
        st.error(f"履歴保存エラー: {e}")

# --------------------------------------------------
# 3. セッション状態の初期化
# --------------------------------------------------
if "history_store" not in st.session_state:
    st.session_state.history_store = load_history_from_file()

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

# 進行ステート・途中経過データの保持用
if "current_step" not in st.session_state:
    st.session_state.current_step = "IDLE"  # "IDLE", "CRITIC_DONE"
if "pending_data" not in st.session_state:
    st.session_state.pending_data = {}

# --------------------------------------------------
# 4. APIキー設定 & サイドバーUI
# --------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ 設定・操作")
    if not api_key:
        api_key = st.text_input("Gemini API Key を入力してください:", type="password")
        st.caption("[Google AI Studio](https://aistudio.google.com/) で取得可能です。")

    st.info(f"使用モデル: **{MODEL_NAME}** (固定)")

    enable_search_option = st.radio(
        "🔍 Web検索 (Grounding) 設定",
        ["自動 (必要時のみ)", "常に有効", "常に無効 (トークン最大節約)"],
        index=0,
        help="無効にするとWeb検索結果の自動追加を停止し、インプットトークンを削減します。"
    )

    st.markdown("---")
    st.subheader("📜 過去の討論履歴")

    history_store = st.session_state.history_store
    options = ["-- 新規作成 / 最新表示 --"]

    sorted_session_ids = sorted(
        history_store.keys(),
        key=lambda k: history_store[k].get("timestamp", ""),
        reverse=True
    )

    for sid in sorted_session_ids:
        item = history_store[sid]
        time_str = item.get("time_label", "")
        topic_preview = item.get("topic", "")[:12]
        options.append(f"[{time_str}] {topic_preview}...")

    selected_option = st.selectbox("閲覧・追記するセッションを選択:", options=options, index=0)

    if selected_option != "-- 新規作成 / 最新表示 --":
        selected_idx = options.index(selected_option) - 1
        st.session_state.active_session_id = sorted_session_ids[selected_idx]
        st.session_state.current_step = "IDLE"
    else:
        if st.session_state.active_session_id not in history_store:
            st.session_state.active_session_id = None

    st.markdown("---")
    if history_store:
        json_string = json.dumps(history_store, ensure_ascii=False, indent=2)
        st.download_button(
            label="💾 全履歴をJSONでバックアップ",
            data=json_string,
            file_name=f"discussion_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    if st.button("🗑️ 画面と全履歴を完全に消去", use_container_width=True):
        st.session_state.history_store = {}
        st.session_state.active_session_id = None
        st.session_state.current_step = "IDLE"
        st.session_state.pending_data = {}
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()

if not api_key:
    st.info("👈 サイドバーから Gemini API キーを設定してください。")
    st.stop()

client = genai.Client(api_key=api_key)

# --------------------------------------------------
# 5. ユーティリティ関数
# --------------------------------------------------
def compress_image_to_bytes(image: Image.Image, max_size=(500, 500)) -> bytes:
    """画像を圧縮してトークンおよびメモリ使用量を削減"""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()

def call_gemini_optimized(
    contents,
    system_instruction: str,
    temperature: float,
    status_container=None,
    use_search: bool = False,
    max_retries: int = 5,
    base_wait: float = 2.0,
    max_wait: float = 60.0
):
    """省トークン・超堅牢リトライ機能付きAPI呼出"""
    current_date = datetime.now().strftime("%Y年%m月%d日")
    context_header = (
        f"【最重要前提】本日は {current_date} です。"
        f"現在、Geminiの最新モデル（Gemini 3.6 Flash等）が利用可能です。"
        f"過去の固定概念に縛られず、常に最新の仕様・ファクトに基づき客観的かつ論理的に思考・出力してください。\n\n"
    )
    full_system = context_header + system_instruction
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None

    config = types.GenerateContentConfig(
        system_instruction=full_system,
        temperature=temperature,
        max_output_tokens=4096,
        tools=tools,
    )

    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
            return res.text or "（応答が空でした。再試行してください。）"
        except APIError as e:
            error_code = getattr(e, "code", None) or getattr(e, "status_code", None)
            error_msg = str(e)
            is_transient = (
                error_code in [429, 500, 502, 503, 504]
                or any(code in error_msg for code in ["429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504"])
            )
            if is_transient:
                if attempt == max_retries - 1:
                    raise e
                exp_wait = min(max_wait, base_wait * (2 ** attempt))
                actual_wait = random.uniform(1.0, exp_wait)
                msg = f"⚠️ API制限/混雑を検知（試行 {attempt+1}/{max_retries}）。{actual_wait:.1f}秒後に自動再試行..."
                if status_container:
                    status_container.warning(msg)
                else:
                    st.warning(msg)
                time.sleep(actual_wait)
                if status_container:
                    status_container.info("🔄 処理を再開します...")
            else:
                raise e
        except Exception as e:
            raise e

def merge_code_continuation(original_text: str, continuation_text: str) -> str:
    """途切れ生成された文章やコードを結合し、Markdown構造を修復"""
    clean_continuation = continuation_text.strip()
    if clean_continuation.startswith("```"):
        lines = clean_continuation.split("\n")
        if len(lines) > 1:
            clean_continuation = "\n".join(lines[1:])

   def merge_code_continuation(original_text: str, continuation_text: str) -> str:
    """途切れ生成された文章やコードを結合し、Markdown構造を修復"""
    clean_continuation = continuation_text.strip()
    if clean_continuation.startswith("```"):
        lines = clean_continuation.split("\n")
        if len(lines) > 1:
            clean_continuation = "\n".join(lines[1:])
            
    return original_text + "\n" + clean_continuation

# --------------------------------------------------
# 6. 状態管理（チャット履歴用）
# --------------------------------------------------
if "debate_messages" not in st.session_state:
    st.session_state.debate_messages = []
if "judge_qa_messages" not in st.session_state:
    st.session_state.judge_qa_messages = []
if "topic" not in st.session_state:
    st.session_state.topic = ""

# --------------------------------------------------
# 7. メイン画面：設定と討論実行
# --------------------------------------------------
st.subheader("🗣️ 討論テーマの設定")
topic_input = st.text_area(
    "討論したいテーマや質問を入力してください:",
    placeholder="例: リモートワークは週5日出社に戻すべきか？",
    height=80
)

# 【要件1】反論役を挟むかどうかの選択スイッチ
route_choice = st.radio(
    "🛤️ 討論の進行ルートを選択してください:",
    [
        "1️⃣ 提案 ➔ 批判 ➔ 🏆審判 (標準・スピーディ)", 
        "2️⃣ 提案 ➔ 批判 ➔ ↩️反論 ➔ 🏆審判 (徹底討論)"
    ],
    index=0,
    horizontal=True
)
include_rebuttal = "徹底討論" in route_choice

# 実行ボタン
if st.button("🚀 討論を開始する", type="primary", use_container_width=True):
    if not topic_input.strip():
        st.warning("⚠️ テーマを入力してください。")
    else:
        # 新規討論のためにステートをリセット
        st.session_state.debate_messages = []
        st.session_state.judge_qa_messages = []
        st.session_state.current_step = "GENERATING"
        st.session_state.topic = topic_input

# --------------------------------------------------
# 8. 討論生成ロジック
# --------------------------------------------------
if st.session_state.current_step == "GENERATING":
    # 検索機能のオンオフ判定（サイドバーの選択を利用）
    use_search_flag = False if "常に無効" in enable_search_option else True

    def add_msg(role, name, text, icon):
        st.session_state.debate_messages.append({"role": role, "name": name, "text": text, "icon": icon})

    st.markdown("---")
    st.info("🔄 討論を進行しています...（少々お待ちください）")
    
    # 1. 提案役のターン
    with st.spinner("💡 提案役が意見を構築中..."):
        prop_sys = "あなたは「提案役」です。与えられたテーマに対し、論理的で説得力のある賛成・推進の立場からの意見を明確に述べてください。"
        prop_res = call_gemini_optimized(st.session_state.topic, prop_sys, 0.7, use_search=use_search_flag)
        add_msg("proposer", "提案役", prop_res, "💡")

    # 2. 批判役のターン
    with st.spinner("⚡ 批判役が問題点を指摘中..."):
        crit_sys = "あなたは「批判役」です。提案役の意見を分析し、隠れたリスク、矛盾、または現実的な問題点を鋭く論理的に指摘してください。"
        crit_prompt = f"【テーマ】: {st.session_state.topic}\n\n【提案役の意見】:\n{prop_res}"
        crit_res = call_gemini_optimized(crit_prompt, crit_sys, 0.7, use_search=use_search_flag)
        add_msg("critic", "批判役", crit_res, "⚡")

    rebuttal_res = ""
    # 3. 反論役のターン（スイッチでONの場合のみ実行）
    if include_rebuttal:
        with st.spinner("↩️ 反論役が再反論を構築中..."):
            reb_sys = "あなたは「反論役」です。批判役の指摘を真摯に受け止めた上で、それを論理的に防御・解決し、提案の価値を再提示してください。"
            reb_prompt = f"【テーマ】: {st.session_state.topic}\n\n【提案】:\n{prop_res}\n\n【批判】:\n{crit_res}"
            rebuttal_res = call_gemini_optimized(reb_prompt, reb_sys, 0.7, use_search=use_search_flag)
            add_msg("rebuttal", "反論役", rebuttal_res, "↩️")

    # 4. 審判役のターン
    with st.spinner("🏆 審判役が最終結論を作成中..."):
        judge_sys = "あなたは「審判役」です。これまでの議論を客観的かつ公平に評価し、最終的な結論と、両者の優れた点・今後の課題を分かりやすくまとめてください。"
        judge_prompt = f"【テーマ】: {st.session_state.topic}\n\n【提案】:\n{prop_res}\n\n【批判】:\n{crit_res}\n\n"
        if include_rebuttal:
            judge_prompt += f"【反論】:\n{rebuttal_res}\n\n"
        judge_res = call_gemini_optimized(judge_prompt, judge_sys, 0.7, use_search=use_search_flag)
        add_msg("judge", "審判役", judge_res, "🏆")

    # 進行ステータスを完了に変更して画面をリロード
    st.session_state.current_step = "JUDGE_DONE"
    st.rerun()

# --------------------------------------------------
# 9. 討論結果の表示と追加質問チャット
# --------------------------------------------------
if st.session_state.current_step == "JUDGE_DONE" and st.session_state.debate_messages:
    st.markdown("---")
    st.subheader("📜 討論記録")
    
    # 討論履歴の描画
    for msg in st.session_state.debate_messages:
        with st.chat_message(msg["role"], avatar=msg["icon"]):
            st.markdown(f"**{msg['name']}**")
            st.write(msg["text"])

    st.markdown("---")
    
    # 【要件2】審判役への追加質問チャット
    st.subheader("❓ 審判役への追加質問")
    st.caption("出された結論について、気になる点や深掘りしたいことを審判役にチャット形式で質問できます。")

    # これまでのQA履歴を表示
    for qa in st.session_state.judge_qa_messages:
        with st.chat_message(qa["role"], avatar=qa["icon"]):
            st.write(qa["text"])

    # チャット入力欄
    if user_q := st.chat_input("審判役へ質問を入力してください..."):
        # ユーザーの質問を保存・表示
        st.session_state.judge_qa_messages.append({"role": "user", "text": user_q, "icon": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.write(user_q)

        # 審判役に質問を投げて回答を生成
        with st.chat_message("judge", avatar="🏆"):
            with st.spinner("審判役が思考中..."):
                qa_sys = (
                    "あなたは先ほどの討論の「審判役」です。自身の出した結論とこれまでの討論内容を踏まえ、"
                    "ユーザーからの追加質問に客観的かつ論理的に答えてください。"
                )
                
                # これまでの文脈を構築
                context = f"【テーマ】{st.session_state.topic}\n\n【討論記録】\n"
                for m in st.session_state.debate_messages:
                    context += f"{m['name']}: {m['text']}\n"
                context += f"\n【ユーザーからの追加質問】\n{user_q}"

                # 回答の生成
                use_search_flag = False if "常に無効" in enable_search_option else True
                qa_res = call_gemini_optimized(context, qa_sys, 0.7, use_search=use_search_flag)
                st.write(qa_res)
                
        # 審判の回答を保存
        st.session_state.judge_qa_messages.append({"role": "judge", "text": qa_res, "icon": "🏆"})
