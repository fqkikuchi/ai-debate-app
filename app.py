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
# 1. ページ初期設定 & 定数
# --------------------------------------------------
st.set_page_config(
    page_title="汎用AIマルチエージェント討論 (Gemini 3.6 Flash高信頼版)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 汎用AIマルチエージェント討論システム")
st.caption(
    "💡提案役 ➔ ⚡批判役 ➔ ↩️反論役 ➔ 🏆審判役 | あらゆる議題に対応する高精度ディベート＆最終成果物自動生成機能"
)

HISTORY_FILE = "discussion_history.json"
MAX_HISTORY_COUNT = 10  # メモリ・ストレージ保護のための最大保持件数

# --------------------------------------------------
# 2. ローカルJSONファイル操作関数 (アトミック書き込み＆堅牢化)
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
    ローカルJSONファイルへ過去ログを書き出す
    - 上限件数制御 (FIFO: 新しい順にMAX_HISTORY_COUNT件を保持)
    - アトミック書き込みによるファイル破損防止
    - メモリ(session_state)との完全同期
    """
    try:
        # 件数制限 (新しい順に並べ替えて最新件数のみ抽出)
        sorted_keys = sorted(
            history_data.keys(),
            key=lambda k: history_data[k].get("timestamp", ""),
            reverse=True
        )
        trimmed_data = {k: history_data[k] for k in sorted_keys[:MAX_HISTORY_COUNT]}

        # 1. メモリ領域 (st.session_state) 側のデータも最新のトリミング状態に更新
        st.session_state.history_store = trimmed_data

        # 2. アトミック書き込み (一時ファイル作成後に置換することで書き込み事故を完全防止)
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
# 4. APIキーとモデル設定 & サイドバー履歴管理
# --------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ 設定・操作")
    if not api_key:
        api_key = st.text_input("Gemini API Key を入力してください:", type="password")
        st.caption("[Google AI Studio](https://aistudio.google.com/) で取得可能です。")

    selected_model = st.selectbox(
        "使用モデル",
        ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash", "カスタム入力"],
        index=0,
    )

    if selected_model == "カスタム入力":
        selected_model = st.text_input("モデル名を入力:", value="gemini-3.6-flash")

    # --- 📜 過去ログ切り替え ---
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

    # バックアップ（ダウンロード）機能の追加
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

# GenAI クライアントの初期化
client = genai.Client(api_key=api_key)

# --------------------------------------------------
# 5. ユーティリティ関数
# --------------------------------------------------
def compress_image_to_bytes(image: Image.Image, max_size=(600, 600)) -> bytes:
    """画像を圧縮してメモリ効率に優れた bytes 形式で保持"""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

def sanitize_input(text: str) -> str:
    """入力文字列のトリム処理"""
    return text.strip()

def call_gemini_with_jitter_retry(
    contents,
    system_instruction: str,
    temperature: float,
    status_container=None,
    use_search: bool = False,
    max_retries: int = 5,
    base_wait: float = 2.0,
    max_wait: float = 60.0
):
    """超堅牢リトライ機能付きAPI呼出"""
    current_date = datetime.now().strftime("%Y年%m月%d日")
    context_header = (
        f"【最重要前提】本日は {current_date} です。"
        f"現在、Geminiの最新モデル（Gemini 3.6 Flash等）が利用可能です。"
        f"過去の固定概念に縛られず、常に最新の仕様・ファクトに基づき客観的かつ論理的に思考・出力してください。\n\n"
    )
    full_system_instruction = context_header + system_instruction
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None

    config = types.GenerateContentConfig(
        system_instruction=full_system_instruction,
        temperature=temperature,
        max_output_tokens=8192,
        tools=tools,
    )

    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=selected_model, contents=contents, config=config
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
    backtick_matches = re.findall(r"```", merged)
    if len(backtick_matches) % 2 != 0:
        merged += "\n```"

    return merged

# --------------------------------------------------
# 6. メインUI (入力領域)
# --------------------------------------------------
topic_raw = st.text_area(
    "検討したいテーマ、ビジネス課題、企画案、意思決定案件、または技術的課題を入力してください:",
    placeholder="例：新規事業として「高齢者向けAI健康管理サービス」を立ち上げる際の参入戦略とリスク対策を議論してください。",
    height=110,
)

uploaded_file = st.file_uploader("📷 関連画像・参考資料（任意）", type=["png", "jpg", "jpeg", "webp"])
image_bytes = None
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    image_bytes = compress_image_to_bytes(raw_image)
    st.image(image_bytes, caption="添付画像（最適化済み）", width=250)

# --------------------------------------------------
# 7. 討論実行ロジック
# --------------------------------------------------
if st.button("🚀 ディベートを開始し、最適解・成果物を生成", type="primary", use_container_width=True):
    topic = sanitize_input(topic_raw)

    if not topic and image_bytes is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        new_session_id = str(uuid.uuid4())
        img_obj = Image.open(io.BytesIO(image_bytes)) if image_bytes else None

        current_res = {}
        try:
            with st.status("🤖 AIエージェントたちが検証・ディベート中...", expanded=True) as status:

                # PHASE 1: 提案役
                status.write("1/4 💡 【提案役】最新情報・論点を調査し、コア提案を構築中...")
                sys_proposer = (
                    "あなたはこの課題を解決へ導く最高レベルの戦略提案役です。"
                    "Google検索を活用し、ユーザーの目的を達成するための最新ファクト・具体的かつ効果的なアプローチを提示してください。\n"
                    "・前置きや挨拶は排除し、1. [コアとなる主張・方針], 2. [最新根拠/ファクト], 3. [具体的推進・実行アプローチ] で記述してください。"
                )
                contents_p1 = [img_obj, f"### 当初の目的と検討テーマ\n{topic}"] if img_obj else [f"### 当初の目的と検討テーマ\n{topic}"]
                current_res["proposer"] = call_gemini_with_jitter_retry(contents_p1, sys_proposer, 0.7, status, use_search=True)

                # PHASE 2: 批判役
                status.write("2/4 ⚡ 【批判役】盲点、リスク、トレードオフ、懸念点を徹底検証中...")
                sys_critic = (
                    "あなたは鋭い観察眼を持つ批判役・リスクアナリストです。"
                    "提案役の意見を踏まえ、Google検索も活用して、提案に含まれる盲点、実現可能性の課題、潜在的リスク、費用対効果や運用の問題点をロジカルに指摘してください。\n"
                    "・前置きは排除し、1. [提案・アプローチの弱点・リスク], 2. [裏付けるファクト/見落とされている現実的課題], 3. [解決困難なボトルネック] で記述してください。"
                )
                contents_p2 = [f"### 当初の目的と検討テーマ\n{topic}\n\n### 提案役の主張\n{current_res['proposer']}"]
                current_res["critic"] = call_gemini_with_jitter_retry(contents_p2, sys_critic, 0.7, status, use_search=True)

                # PHASE 3: 反論役
                status.write("3/4 ↩️ 【反論役】批判を克服する改善策・現実的補強策を策定中...")
                sys_rebutter = (
                    "あなたは提案役をサポートし、批判を克服する反論役です。"
                    "批判役の指摘を建設的に分析し、リスクを最小化・回避するための具体的な改善策やピボット案（軌道修正案）を提示してください。\n"
                    "・前置きは排除し、1. [批判の受容とピボット案], 2. [リスクを軽減する具体的な代替・補強アプローチ], 3. [成果物完成に向けた具体策] で記述してください。"
                )
                contents_p3 = [f"### 当初の目的と検討テーマ\n{topic}\n\n### 提案役の主張\n{current_res['proposer']}\n\n### 批判役の指摘\n{current_res['critic']}"]
                current_res["rebutter"] = call_gemini_with_jitter_retry(contents_p3, sys_rebutter, 0.4, status, use_search=False)

                # PHASE 4: 審判役
                status.write("4/4 🏆 【審判役】議論を総合評価し、最終成果物を出力中...")
                sys_judge = (
                    "あなたは最高権威の審判役兼ファシリテーターです。これまでの議論を総合評価し、ユーザーが求める最終成果物を出力してください。\n\n"
                    "【絶対遵守ルール】\n"
                    "1. 単なる抽象的アドバイスで終わらせず、ユーザーの要望に応じた『即実行・実用可能な完全版成果物』を作成してください。\n"
                    "2. 成果物にコードが含まれる場合は「...（省略）」等を一切使わず、そのまま動く完全なコードを提示してください。\n\n"
                    "【出力構成】\n"
                    "1. [議論サマリーと最終判定 (Go / No-Go / 条件付きGo)]\n"
                    "2. [採用した最終方針・主要リスクへの対策]\n"
                    "3. [完成版成果物（そのまま活用できる具体的な文章・コード・計画等）]"
                )
                contents_p4 = [f"### 【ユーザーの当初の目的・要望】\n{topic}\n\n### 【提案役の主張】\n{current_res['proposer']}\n\n### 【批判役の指摘】\n{current_res['critic']}\n\n### 【反論・解決案】\n{current_res['rebutter']}\n\n--- 命令 ---\n上記の議論成果を統合し、ユーザーの要求を満たす最終成果物を出力してください。"]
                current_res["judge"] = call_gemini_with_jitter_retry(contents_p4, sys_judge, 0.2, status, use_search=False)

                # セッション構造の作成とJSON保存
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
# 8. 討論結果 UI (折りたたみ表示 & 各種機能)
# --------------------------------------------------
active_id = st.session_state.active_session_id
active_data = st.session_state.history_store.get(active_id) if active_id else None

if active_data:
    res = active_data.get("result", {})
    topic_used = active_data.get("topic", "")

    st.markdown("---")
    st.header(f"🗣️ AIディベートプロセス (テーマ: {topic_used[:30]}...)")
    st.caption("💡 スクロール量を削減するため、提案・批判・反論はデフォルトで折りたたんでいます。タップで開閉可能です。")

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
                        "あなたは審判役です。直前の出力が途切れたため、続きのみを出力してください。\n"
                        f"【直前の出力末尾】\n{last_context}"
                    )
                    contents_cont = [f"### 直前の全体出力\n{res['judge']}"]
                    res_cont = call_gemini_with_jitter_retry(contents_cont, sys_continue, 0.1, status)
                    merged = merge_code_continuation(res["judge"], res_cont)

                    st.session_state.history_store[active_id]["result"]["judge"] = merged
                    save_history_to_file(st.session_state.history_store)
                    status.update(label="✅ 補完完了！", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")

        # --------------------------------------------------
        # 9. 追加質問・QA機能（マルチターン文脈維持）
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
                        "これまでのディベート内容および「過去の質疑応答文脈」をすべて考慮した上で、ユーザーの追加指示に応えてください。\n"
                        "※成果物の修正・改訂を求められた場合は、途中で省略せず完全な改訂成果物を提示してください。"
                    )

                    history_context_str = ""
                    for h in chat_history[:-1]:
                        role_label = "ユーザー" if h["role"] == "user" else "審判役"
                        history_context_str += f"\n【{role_label}】: {h['content']}\n"

                    prompt_contents = [
                        f"### 当初のテーマ\n{topic_used}\n\n"
                        f"### 審判役の初期成果物\n{res['judge']}\n\n"
                        f"### 過去の追加質疑応答の文脈\n{history_context_str if history_context_str else '（過去の質疑応答はありません）'}\n\n"
                        f"### 今回のユーザーの指示・質問\n{user_query}"
                    ]

                    try:
                        ans_text = call_gemini_with_jitter_retry(prompt_contents, sys_chat_judge, 0.2)
                        st.markdown(ans_text)
                        chat_history.append({"role": "assistant", "content": ans_text})

                        st.session_state.history_store[active_id]["chat_history"] = chat_history
                        save_history_to_file(st.session_state.history_store)
                    except Exception as e:
                        st.error(f"回答生成エラー: {e}")
