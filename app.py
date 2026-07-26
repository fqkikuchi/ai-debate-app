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
    "💡提案役 ➔ ⚡批判役 ➔ ↩️反論役 ➔ 🏆審判役 | Gemini 3.6 Flash専用・省トークン最適化版"
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
    """
    ローカルJSONファイルへ過去ログを書き出す（アトミック書き込み）
    """
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

    # 検索機能の明示的制御（トークン節約設定）
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

    merged = original_text.rstrip() + "\n" + clean_continuation
    if len(re.findall(r"```", merged)) % 2 != 0:
        merged += "\n```"

    return merged

# --------------------------------------------------
# 6. メインUI (入力領域)
# --------------------------------------------------
topic_raw = st.text_area(
    "検討したいテーマ、ビジネス課題、企画案、意思決定案件を入力してください:",
    placeholder="例：新規事業として「高齢者向けAI健康管理サービス」を立ち上げる際の参入戦略とリスク対策を議論してください。",
    height=110,
)

uploaded_file = st.file_uploader("📷 関連画像・参考資料（任意）", type=["png", "jpg", "jpeg", "webp"])
image_bytes = None
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    image_bytes = compress_image_to_bytes(raw_image)
    st.image(image_bytes, caption="添付画像（軽量化済み）", width=220)

# --------------------------------------------------
# 7. 討論実行ロジック (省トークン・高効率処理)
# --------------------------------------------------
if st.button("🚀 ディベートを開始し、最適解・成果物を生成", type="primary", use_container_width=True):
    topic = topic_raw.strip()

    if not topic and image_bytes is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        new_session_id = str(uuid.uuid4())
        img_obj = Image.open(io.BytesIO(image_bytes)) if image_bytes else None

        # 検索実施の条件判定
        if enable_search_option == "常に有効":
            should_search = True
        elif enable_search_option == "常に無効 (トークン最大節約)":
            should_search = False
        else:
            # 自動判定: 特定キーワードが含まれる場合のみ検索ONとし基本はOFF
            search_keywords = ["最新", "202", "ニュース", "競合", "法改正", "トレンド", "株価", "市場規模", "動向", "現在"]
            should_search = any(kw in topic for kw in search_keywords)

        current_res = {}
        try:
            with st.status("🤖 AIエージェントたちが検証・ディベート中...", expanded=True) as status:

                # PHASE 1: 提案役 (条件付き検索)
                status.write("1/4 💡 【提案役】コア提案を構築中...")
                sys_proposer = (
                    "あなたはこの課題を解決へ導く最高レベルの戦略提案役です。"
                    "挨拶や不要な前置きを完全に排除し、簡潔かつ論理的に記述してください。\n"
                    "1. [コアとなる主張・方針], 2. [根拠/ファクト], 3. [具体的推進アプローチ] (箇条書き推奨)"
                )
                contents_p1 = [img_obj, f"### テーマ\n{topic}"] if img_obj else [f"### テーマ\n{topic}"]
                current_res["proposer"] = call_gemini_optimized(
                    contents_p1, sys_proposer, 0.6, status, use_search=should_search
                )

                # PHASE 2: 批判役 (条件付き検索)
                status.write("2/4 ⚡ 【批判役】リスク・盲点を検証中...")
                sys_critic = (
                    "あなたは鋭い批判役・リスクアナリストです。"
                    "前置きを排除し、提案の弱点や現実的リスクを箇条書きで厳格に指摘してください。\n"
                    "1. [提案の弱点・盲点], 2. [実現性の課題・ボトルネック], 3. [費用対効果・運用の問題]"
                )
                contents_p2 = [f"### テーマ\n{topic}\n\n### 提案役の主張\n{current_res['proposer']}"]
                current_res["critic"] = call_gemini_optimized(
                    contents_p2, sys_critic, 0.6, status, use_search=should_search
                )

                # PHASE 3: 反論役 (検索OFF固定)
                status.write("3/4 ↩️ 【反論役】批判を克服する改善策を策定中...")
                sys_rebutter = (
                    "あなたは提案役をサポートし、批判を克服する反論役です。"
                    "批判を受け入れつつ、リスクを回避・軽減するピボット案と具体的補強策を提示してください。\n"
                    "1. [批判の受容とピボット案], 2. [リスク軽減策], 3. [成果物補強策]"
                )
                contents_p3 = [
                    f"### テーマ\n{topic}\n\n### 提案\n{current_res['proposer']}\n\n### 批判\n{current_res['critic']}"
                ]
                current_res["rebutter"] = call_gemini_optimized(
                    contents_p3, sys_rebutter, 0.4, status, use_search=False
                )

                # PHASE 4: 審判役 (検索OFF固定・最終成果物作成)
                status.write("4/4 🏆 【審判役】最終判定および成果物を生成中...")
                sys_judge = (
                    "あなたは最高権威の審判役兼ファシリテーターです。これまでの議論を統合し、ユーザーの要望を満たす『即実行・実用可能な完全版成果物』を出力してください。\n"
                    "コードが含まれる場合は省略せず完全なコードを提示してください。\n\n"
                    "【出力構成】\n"
                    "1. [議論サマリーと最終判定 (Go / No-Go / 条件付きGo)]\n"
                    "2. [採用した最終方針・主要リスクへの対策]\n"
                    "3. [完成版成果物（そのまま活用できる具体的な文章・コード・計画等）]"
                )
                contents_p4 = [
                    f"### 【ユーザーの要望】\n{topic}\n\n"
                    f"### 【提案】\n{current_res['proposer']}\n\n"
                    f"### 【批判】\n{current_res['critic']}\n\n"
                    f"### 【反論・改善案】\n{current_res['rebutter']}"
                ]
                current_res["judge"] = call_gemini_optimized(
                    contents_p4, sys_judge, 0.2, status, use_search=False
                )

                # 保存処理
                now_dt = datetime.now()
                session_data = {
                    "timestamp": now_dt.isoformat(),
                    "time_label": now_dt.strftime("%H:%M:%S"),
                    "topic": topic,
                    "result": current_res,
                    "chat_history": []
                }

                st.session_state.history_store[new_session_id] = session_data
                st.session_state.active_session_id = new_session_id
                save_history_to_file(st.session_state.history_store)

                status.update(label="✅ 討論および最終成果物の生成が完了しました！", state="complete", expanded=False)

        except Exception as e:
            st.error(f"実行中にエラーが発生しました: {e}")

# --------------------------------------------------
# 8. 討論結果 UI 表示
# --------------------------------------------------
active_id = st.session_state.active_session_id
active_data = st.session_state.history_store.get(active_id) if active_id else None

if active_data:
    res = active_data.get("result", {})
    topic_used = active_data.get("topic", "")

    st.markdown("---")
    st.header(f"🗣️ AIディベートプロセス (テーマ: {topic_used[:30]}...)")

    if "proposer" in res:
        with st.expander("1. 💡 【提案役】の主張を見る", expanded=False):
            st.markdown(res["proposer"])

    if "critic" in res:
        with st.expander("2. ⚡ 【批判役】の指摘・リスク検証を見る", expanded=False):
            st.markdown(res["critic"])

    if "rebutter" in res:
        with st.expander("3. ↩️ 【反論役】の反論・改善策を見る", expanded=False):
            st.markdown(res["rebutter"])

    if "judge" in res:
        st.markdown("---")
        st.header("🏆 審判役の最終結論・最終成果物")

        with st.container(border=True):
            st.markdown(res["judge"])

        if st.button("📝 文章や成果物が途中で切れている場合、続きを生成して自動結合する"):
            try:
                with st.status("🔄 審判役の続きを生成し、構文修復中...", expanded=True) as status:
                    last_context = res["judge"][-600:] if len(res["judge"]) > 600 else res["judge"]
                    sys_continue = (
                        "あなたは審判役です。直前の出力が途切れたため、続きのみを簡潔に出力してください。\n"
                        f"【直前の出力末尾】\n{last_context}"
                    )
                    contents_cont = [f"### 直前の全体出力\n{res['judge']}"]
                    res_cont = call_gemini_optimized(contents_cont, sys_continue, 0.1, status, use_search=False)
                    merged = merge_code_continuation(res["judge"], res_cont)

                    st.session_state.history_store[active_id]["result"]["judge"] = merged
                    save_history_to_file(st.session_state.history_store)
                    status.update(label="✅ 補完完了！", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")

        # --------------------------------------------------
        # 9. 追加質問・QA機能（スライドウィンドウ方式）
        # --------------------------------------------------
        st.markdown("---")
        st.header("💬 審判役への追加質問・修正指示")

        chat_history = active_data.get("chat_history", [])

        for chat in chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        user_query = st.chat_input("審判役に指示や追加質問を入力...")

        if user_query:
            chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("🤖 これまでの文脈を踏まえて回答生成中..."):
                    sys_chat_judge = (
                        "あなたは最高権威の審判役兼ファシリテーターです。\n"
                        "初期成果物および「直近の質疑応答」を踏まえ、ユーザーの指示に応えてください。\n"
                        "成果物の修正を求められた場合は省略せず完全な改訂成果物を提示してください。"
                    )

                    # 直近3ターン（ユーザー・アシスタント最大6件）のみ抽出してインプットトークンを削減
                    recent_chats = chat_history[-6:-1]
                    history_context_str = ""
                    for h in recent_chats:
                        role_label = "ユーザー" if h["role"] == "user" else "審判役"
                        history_context_str += f"\n【{role_label}】: {h['content']}\n"

                    # 審判役成果物の冒頭・重要文脈のみを送信し節約
                    judge_summary_context = res['judge'][:1200] + "\n...(以下省略)" if len(res['judge']) > 1200 else res['judge']

                    prompt_contents = [
                        f"### 当初のテーマ\n{topic_used}\n\n"
                        f"### 審判役の成果物（要約/抜粋）\n{judge_summary_context}\n\n"
                        f"### 直近の質疑応答\n{history_context_str if history_context_str else '（なし）'}\n\n"
                        f"### 今回のユーザーの指示・質問\n{user_query}"
                    ]

                    try:
                        ans_text = call_gemini_optimized(prompt_contents, sys_chat_judge, 0.2, use_search=False)
                        st.markdown(ans_text)
                        chat_history.append({"role": "assistant", "content": ans_text})

                        st.session_state.history_store[active_id]["chat_history"] = chat_history
                        save_history_to_file(st.session_state.history_store)
                    except Exception as e:
                        st.error(f"回答生成エラー: {e}")
