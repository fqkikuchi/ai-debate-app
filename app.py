from datetime import datetime
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
    "💡提案役 ➔ ⚡批判役 ➔ ⏸️(ユーザー選択) ➔ ↩️反論役(任意) ➔ 🏆審判役 | Gemini 3.6 Flash専用・省トークン版"
)

HISTORY_FILE = "discussion_history.json"
MAX_HISTORY_COUNT = 10
MODEL_NAME = "gemini-3.6-flash"  # Gemini 3.6 Flashに完全固定

# --------------------------------------------------
# 2. ローカルJSONファイル操作関数 (アトミック書き込み)
# --------------------------------------------------
def load_history_from_file() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"履歴読み込みエラー: {e}")
            return {}
    return {}

def save_history_to_file(history_data: dict):
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

if "current_step" not in st.session_state:
    st.session_state.current_step = "IDLE"
if "pending_data" not in st.session_state:
    st.session_state.pending_data = {}
if "debate_messages" not in st.session_state:
    st.session_state.debate_messages = []
if "judge_qa_messages" not in st.session_state:
    st.session_state.judge_qa_messages = []
if "topic" not in st.session_state:
    st.session_state.topic = ""

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
        index=0
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
    if st.button("🗑️ 画面と全履歴を完全に消去", use_container_width=True):
        st.session_state.history_store = {}
        st.session_state.active_session_id = None
        st.session_state.current_step = "IDLE"
        st.session_state.pending_data = {}
        st.session_state.debate_messages = []
        st.session_state.judge_qa_messages = []
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
def call_gemini_optimized(contents, system_instruction: str, temperature: float, use_search: bool = False, max_retries: int = 5):
    """省トークン・超堅牢リトライ機能付きAPI呼出"""
    current_date = datetime.now().strftime("%Y年%m月%d日")
    full_system = (
        f"【最重要前提】本日は {current_date} です。最新の仕様・ファクトに基づき客観的かつ論理的に思考してください。\n\n"
        + system_instruction
    )
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None
    config = types.GenerateContentConfig(system_instruction=full_system, temperature=temperature, max_output_tokens=4096, tools=tools)

    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
            return res.text or "（応答が空でした。再試行してください。）"
        except APIError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(random.uniform(1.0, 2.0 * (2 ** attempt)))
        except Exception as e:
            raise e

def add_msg(role, name, text, icon):
    st.session_state.debate_messages.append({"role": role, "name": name, "text": text, "icon": icon})

def get_discussion_context():
    """これまでの討論内容をプロンプト用に結合する関数"""
    context = f"【テーマ】: {st.session_state.topic}\n\n"
    for msg in st.session_state.debate_messages:
        context += f"【{msg['name']}の意見】:\n{msg['text']}\n\n"
    return context

# --------------------------------------------------
# 6. メイン画面：入力フォーム
# --------------------------------------------------
st.subheader("🗣️ 討論テーマの設定")
topic_input = st.text_area(
    "討論したいテーマや質問を入力してください:",
    value=st.session_state.topic if st.session_state.topic else "",
    placeholder="例: リモートワークは週5日出社に戻すべきか？\n例: Pythonで顧客管理アプリを作る場合の実装方針を議論して",
    height=80,
    disabled=(st.session_state.current_step != "IDLE")
)

# 開始ボタン (IDLE状態の時のみ表示)
if st.session_state.current_step == "IDLE":
    if st.button("🚀 討論を開始する (提案 ➔ 批判 まで生成)", type="primary", use_container_width=True):
        if not topic_input.strip():
            st.warning("⚠️ テーマを入力してください。")
        else:
            st.session_state.topic = topic_input
            st.session_state.debate_messages = []
            st.session_state.judge_qa_messages = []
            st.session_state.current_step = "GENERATING_PHASE1"
            st.rerun()

# --------------------------------------------------
# 7. 討論記録の表示 (共通UI)
# --------------------------------------------------
if st.session_state.current_step != "IDLE":
    st.markdown("---")
    st.subheader("📜 討論記録")
    
    for msg in st.session_state.debate_messages:
        with st.chat_message(msg["role"], avatar=msg["icon"]):
            st.markdown(f"**{msg['name']}**")
            st.write(msg["text"])

# --------------------------------------------------
# 8. 動的ステップ実行ロジック
# --------------------------------------------------
use_search_flag = False if "常に無効" in enable_search_option else True

# 【STEP 1】: 提案役と批判役の生成
if st.session_state.current_step == "GENERATING_PHASE1":
    with st.spinner("💡 提案役が意見を構築中..."):
        prop_sys = "あなたは「提案役」です。与えられたテーマに対し、論理的で説得力のある賛成・推進の立場からの意見を明確に述べてください。"
        prop_res = call_gemini_optimized(f"【テーマ】: {st.session_state.topic}", prop_sys, 0.7, use_search=use_search_flag)
        add_msg("proposer", "提案役", prop_res, "💡")

    with st.spinner("⚡ 批判役が問題点を指摘中..."):
        crit_sys = "あなたは「批判役」です。提案役の意見を分析し、隠れたリスク、矛盾、または現実的な問題点を鋭く論理的に指摘してください。"
        crit_res = call_gemini_optimized(get_discussion_context(), crit_sys, 0.7, use_search=use_search_flag)
        add_msg("critic", "批判役", crit_res, "⚡")

    st.session_state.current_step = "WAITING_FOR_USER_CHOICE"
    st.rerun()

# 【STEP 2】: ユーザーのルート選択待ち
elif st.session_state.current_step == "WAITING_FOR_USER_CHOICE":
    st.markdown("---")
    st.info("💡 提案役と批判役の意見が出揃いました。ここまでの議論を踏まえて、次に行うアクションを選択してください。")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ 反論役を追加する (徹底討論)", use_container_width=True):
            st.session_state.current_step = "GENERATING_REBUTTAL"
            st.rerun()
    with col2:
        if st.button("🏆 審判役に結論を出させる (ここで終了)", type="primary", use_container_width=True):
            st.session_state.current_step = "GENERATING_JUDGE"
            st.rerun()

# 【STEP 3】: 反論役の生成 (選択時のみ)
elif st.session_state.current_step == "GENERATING_REBUTTAL":
    st.markdown("---")
    with st.spinner("↩️ 反論役が再反論を構築中..."):
        reb_sys = "あなたは「反論役」です。批判役の指摘を真摯に受け止めた上で、それを論理的に防御・解決し、提案の価値を再提示してください。"
        rebuttal_res = call_gemini_optimized(get_discussion_context(), reb_sys, 0.7, use_search=use_search_flag)
        add_msg("rebuttal", "反論役", rebuttal_res, "↩️")
        
    st.session_state.current_step = "GENERATING_JUDGE"
    st.rerun()

# 【STEP 4】: 審判役の生成
elif st.session_state.current_step == "GENERATING_JUDGE":
    st.markdown("---")
    with st.spinner("🏆 審判役が最終結論を作成中..."):
        judge_sys = "あなたは「審判役」です。これまでの議論を客観的かつ公平に評価し、最終的な結論と、両者の優れた点・今後の課題を分かりやすくまとめてください。"
        judge_res = call_gemini_optimized(get_discussion_context(), judge_sys, 0.7, use_search=use_search_flag)
        add_msg("judge", "審判役", judge_res, "🏆")

    st.session_state.current_step = "JUDGE_DONE"

    # --- 履歴の保存処理をここに追加 ---
    if st.session_state.active_session_id is None:
        st.session_state.active_session_id = str(uuid.uuid4())
        
    session_id = st.session_state.active_session_id
    st.session_state.history_store[session_id] = {
        "timestamp": datetime.now().isoformat(),
        "time_label": datetime.now().strftime("%m/%d %H:%M"),
        "topic": st.session_state.topic,
        "messages": st.session_state.debate_messages
    }
    save_history_to_file(st.session_state.history_store)
    # ---------------------------------

    st.rerun()

# --------------------------------------------------
# 9. 審判役への追加質問・最終出力依頼
# --------------------------------------------------
if st.session_state.current_step == "JUDGE_DONE":
    st.markdown("---")
    st.subheader("❓ 審判役への追加質問・最終成果物の依頼")
    st.caption("結論に対する深掘り質問のほか、「議論を踏まえた最適なコードを書いて」「ドキュメント形式でまとめて出力して」といった作業依頼も可能です。")

    for qa in st.session_state.judge_qa_messages:
        with st.chat_message(qa["role"], avatar=qa["icon"]):
            st.write(qa["text"])

    if user_q := st.chat_input("審判役に質問や作成依頼を入力してください... (例: 議論の結果をもとにコードを生成して)"):
        st.session_state.judge_qa_messages.append({"role": "user", "text": user_q, "icon": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.write(user_q)

        with st.chat_message("judge", avatar="🏆"):
            with st.spinner("審判役が対応中..."):
                qa_sys = (
                    "あなたは先ほどの討論の「審判役」です。自身の出した結論とこれまでの討論内容を踏まえ、"
                    "ユーザーからの追加質問や作業依頼に対応してください。\n"
                    "「コードを書いて」「ドキュメント形式でまとめて」などの要求があった場合は、"
                    "これまでの議論の経緯や結論の要素を最大限に反映し、指定されたフォーマットで高精度な出力を行ってください。"
                )
                qa_prompt = get_discussion_context() + f"【ユーザーからの追加質問・作成依頼】\n{user_q}"
                
                qa_res = call_gemini_optimized(qa_prompt, qa_sys, 0.7, use_search=use_search_flag)
                st.write(qa_res)
                
        st.session_state.judge_qa_messages.append({"role": "judge", "text": qa_res, "icon": "🏆"})
