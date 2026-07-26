from datetime import datetime
import io
import os
import random
import re
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
import streamlit as st

# --------------------------------------------------
# 1. ページ初期設定
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

# --------------------------------------------------
# 2. APIキーとモデル設定 (サイドバー)
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

    st.markdown("---")
    if st.button("🗑️ 画面と全履歴をリセット", use_container_width=True):
        st.session_state.discussion_result = {}
        st.session_state.topic_used = ""
        st.session_state.saved_image_bytes = None
        st.session_state.chat_history = []
        st.rerun()

if not api_key:
    st.info("👈 サイドバーから Gemini API キーを設定してください。")
    st.stop()

# GenAI クライアントの初期化
client = genai.Client(api_key=api_key)

# セッション状態の初期化
if "discussion_result" not in st.session_state:
    st.session_state.discussion_result = {}
if "topic_used" not in st.session_state:
    st.session_state.topic_used = ""
if "saved_image_bytes" not in st.session_state:
    st.session_state.saved_image_bytes = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# 3. ユーティリティ関数（メモリ最適化・堅牢エラー処理・自動結合）
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
    """属性チェックと文字列検索を併用した超堅牢リトライ機能付きAPI呼出"""

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

            # ステータスコード（429, 5xx）またはエラーメッセージから一時エラー判定
            is_transient = (
                error_code in [429, 500, 502, 503, 504]
                or any(code in error_msg for code in ["429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504"])
            )

            if is_transient:
                if attempt == max_retries - 1:
                    raise e
                exp_wait = min(max_wait, base_wait * (2 ** attempt))
                actual_wait = random.uniform(1.0, exp_wait)

                msg = f"⚠️ API混雑/制限を検知（試行 {attempt+1}/{max_retries}）。{actual_wait:.1f}秒後に自動リトライします..."
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

    # 続きのテキストが重複してコードブロック宣言から始まっている場合を除去
    if clean_continuation.startswith("```"):
        lines = clean_continuation.split("\n")
        if len(lines) > 1:
            clean_continuation = "\n".join(lines[1:])

    merged = original_text.rstrip() + "\n" + clean_continuation

    # コードブロック（```）の閉鎖判定と自動補正
    backtick_matches = re.findall(r"```", merged)
    if len(backtick_matches) % 2 != 0:
        merged += "\n```"

    return merged

# --------------------------------------------------
# 4. メインUI (入力領域)
# --------------------------------------------------
topic_raw = st.text_area(
    "検討したいテーマ、ビジネス課題、企画案、意思決定案件、または技術的課題を入力してください:",
    placeholder="例1：新規事業として「高齢者向けAI健康管理サービス」を立ち上げる際の参入戦略とリスク対策を議論してください。\n"
                "例2：リモートワークと出社ハイブリッド制度における生産性低下を防ぐための社内運用ルール案を作成してください。\n"
                "例3：PythonでWebスクレイピングを行い、データをCSV保存するスクリプトを作成してください。",
    height=140,
)

uploaded_file = st.file_uploader(
    "📷 関連画像・参考資料（任意）",
    type=["png", "jpg", "jpeg", "webp"],
)

image_bytes = None
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    image_bytes = compress_image_to_bytes(raw_image)
    st.image(image_bytes, caption="添付画像（最適化済み）", width=250)

# --------------------------------------------------
# 5. 討論実行ロジック (段階的保存・状態保持対応)
# --------------------------------------------------
if st.button("🚀 ディベートを開始し、最適解・成果物を生成", type="primary", use_container_width=True):
    topic = sanitize_input(topic_raw)

    if not topic and image_bytes is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        st.session_state.topic_used = topic
        st.session_state.saved_image_bytes = image_bytes
        st.session_state.discussion_result = {}
        st.session_state.chat_history = []

        # API送信用画像オブジェクトの一時作成
        img_obj = Image.open(io.BytesIO(image_bytes)) if image_bytes else None

        try:
            with st.status("🤖 AIエージェントたちが検証・ディベート中...", expanded=True) as status:

                # --- PHASE 1: 提案役 ---
                status.write("1/4 💡 【提案役】最新情報・論点を調査し、コア提案を構築中...")
                sys_proposer = (
                    "あなたはこの課題を解決へ導く最高レベルの戦略提案役です。"
                    "Google検索を活用し、ユーザーの目的を達成するための最新ファクト・具体的かつ効果的なアプローチを提示してください。\n"
                    "・前置きや挨拶は排除し、1. [コアとなる主張・方針], 2. [最新根拠/ファクト], 3. [具体的推進・実行アプローチ] で記述してください。"
                )
                contents_p1 = [img_obj, f"### 当初の目的と検討テーマ\n{topic}"] if img_obj else [f"### 当初の目的と検討テーマ\n{topic}"]
                res_p1 = call_gemini_with_jitter_retry(contents_p1, sys_proposer, 0.7, status, use_search=True)
                st.session_state.discussion_result["proposer"] = res_p1

                # --- PHASE 2: 批判役 ---
                status.write("2/4 ⚡ 【批判役】盲点、リスク、トレードオフ、懸念点を徹底検証中...")
                sys_critic = (
                    "あなたは鋭い観察眼を持つ批判役・リスクアナリストです。"
                    "提案役の意見を踏まえ、Google検索も活用して、提案に含まれる盲点、実現可能性の課題、潜在的リスク、費用対効果や運用の問題点をロジカルに指摘してください。\n"
                    "・前置きは排除し、1. [提案・アプローチの弱点・リスク], 2. [裏付けるファクト/見落とされている現実的課題], 3. [解決困難なボトルネック] で記述してください。"
                )
                contents_p2 = [
                    f"### 当初の目的と検討テーマ\n{topic}\n\n"
                    f"### 提案役の主張\n{st.session_state.discussion_result['proposer']}"
                ]
                res_p2 = call_gemini_with_jitter_retry(contents_p2, sys_critic, 0.7, status, use_search=True)
                st.session_state.discussion_result["critic"] = res_p2

                # --- PHASE 3: 反論役 ---
                status.write("3/4 ↩️ 【反論役】批判を克服する改善策・現実的補強策を策定中...")
                sys_rebutter = (
                    "あなたは提案役をサポートし、批判を克服する反論役です。"
                    "批判役の指摘を建設的に分析し、リスクを最小化・回避するための具体的な改善策やピボット案（軌道修正案）を提示してください。\n"
                    "・前置きは排除し、1. [批判の受容とピボット案], 2. [リスクを軽減する具体的な代替・補強アプローチ], 3. [成果物完成に向けた具体策] で記述してください。"
                )
                contents_p3 = [
                    f"### 当初の目的と検討テーマ\n{topic}\n\n"
                    f"### 提案役の主張\n{st.session_state.discussion_result['proposer']}\n\n"
                    f"### 批判役の指摘\n{st.session_state.discussion_result['critic']}"
                ]
                res_p3 = call_gemini_with_jitter_retry(contents_p3, sys_rebutter, 0.4, status, use_search=False)
                st.session_state.discussion_result["rebutter"] = res_p3

                # --- PHASE 4: 審判役 ---
                status.write("4/4 🏆 【審判役】議論を総合評価し、最終成果物を出力中...")
                sys_judge = (
                    "あなたは最高権威の審判役兼ファシリテーターです。これまでの議論を総合評価し、ユーザーが求める最終成果物を出力してください。\n\n"
                    "【絶対遵守ルール】\n"
                    "1. 単なる抽象的アドバイスで終わらせず、ユーザーの要望に応じた『即実行・実用可能な完全版成果物（企画書案、マニュアル、具体的対案、またはプログラムコード等）』を作成してください。\n"
                    "2. 成果物にコードが含まれる場合は「...（省略）」等を一切使わず、そのまま動く完全なコードを提示してください。文章の場合も途中で切り上げず完成された形式にしてください。\n\n"
                    "【出力構成】\n"
                    "1. [議論サマリーと最終判定 (Go / No-Go / 条件付きGo)]\n"
                    "2. [採用した最終方針・主要リスクへの対策]\n"
                    "3. [完成版成果物（そのまま活用できる具体的な文章・コード・計画等）]"
                )
                contents_p4 = [
                    f"### 【ユーザーの当初の目的・要望】\n{topic}\n\n"
                    f"### 【提案役の主張】\n{st.session_state.discussion_result['proposer']}\n\n"
                    f"### 【批判役の指摘】\n{st.session_state.discussion_result['critic']}\n\n"
                    f"### 【反論・解決案】\n{st.session_state.discussion_result['rebutter']}\n\n"
                    f"--- 命令 ---\n"
                    f"上記の議論成果を統合し、ユーザーの要求を満たす最終成果物を出力してください。"
                ]
                res_p4 = call_gemini_with_jitter_retry(contents_p4, sys_judge, 0.2, status, use_search=False)
                st.session_state.discussion_result["judge"] = res_p4

                status.update(label="✅ 討論および最終成果物の生成が完了しました！", state="complete", expanded=False)

        except Exception as e:
            st.error(f"実行中にエラーが発生しました: {e}")
            st.info("💡 途中のフェーズまで実行結果が生成されている場合、下部に表示されています。")

# --------------------------------------------------
# 6. 討論結果のUI描画（完全時系列順表示：提案 ➔ 批判 ➔ 反論 ➔ 審判）
# --------------------------------------------------
res = st.session_state.discussion_result

if res:
    # --- 議論プロセスの表示 (時系列順: 1.提案 → 2.批判 → 3.反論) ---
    st.markdown("---")
    st.header("🗣️ AIディベートプロセス (時系列)")

    if "proposer" in res:
        with st.expander("1. 💡 【提案役】の主張", expanded=True):
            st.markdown(res["proposer"])

    if "critic" in res:
        with st.expander("2. ⚡ 【批判役】の指摘・リスク検証", expanded=True):
            st.markdown(res["critic"])

    if "rebutter" in res:
        with st.expander("3. ↩️ 【反論役】の反論・改善策", expanded=True):
            st.markdown(res["rebutter"])

    # --- 審判役の最終結論および完全成果物 ---
    if "judge" in res:
        st.markdown("---")
        st.header("🏆 審判役の最終結論・最終成果物")
        st.info("💡 議論を統合した最終成果物です。そのまま活用・コピーできる内容が出力されます。")

        with st.container(border=True):
            st.markdown(res["judge"])

        # テキスト・コード途切れ時の補完UI
        if st.button("📝 文章や成果物が途中で切れている場合、続きを生成して自動結合する"):
            try:
                with st.status("🔄 審判役の続きを生成し、構文修復中...", expanded=True) as status:
                    last_context = res["judge"][-600:] if len(res["judge"]) > 600 else res["judge"]

                    sys_continue = (
                        "あなたは審判役です。あなたの直前の出力がトークン制限により途中で切れてしまいました。\n"
                        "以下の【直前の出力末尾】から自然につながるように、途切れたテキストやコードの「完全な続き」のみを出力してください。\n"
                        "※前置き、挨拶、重複文節は絶対に出力しないでください。\n\n"
                        f"【直前の出力末尾】\n{last_context}"
                    )

                    contents_cont = [
                        f"### 当初の目的・要望\n{st.session_state.topic_used}\n\n"
                        f"### 直前までの全体出力\n{res['judge']}"
                    ]

                    res_cont = call_gemini_with_jitter_retry(contents_cont, sys_continue, 0.1, status, use_search=False)
                    merged_result = merge_code_continuation(st.session_state.discussion_result["judge"], res_cont)
                    st.session_state.discussion_result["judge"] = merged_result

                    status.update(label="✅ 続きの生成と構造の修復が完了しました！", state="complete")

                st.rerun()

            except Exception as e:
                st.error(f"続きの生成中にエラーが発生しました: {e}")

        # --------------------------------------------------
        # 7. 審判役に対する追加質問・対話機能 (QAチャット)
        # --------------------------------------------------
        st.markdown("---")
        st.header("💬 審判役への追加質問・条件変更")
        st.caption("審判役の結論や成果物に対して、追加の要望や修正指示（「要約して」「別の視点も追加して」「コード化して」など）を質疑応答できます。")

        # 過去のチャット履歴表示
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        # チャット入力
        user_query = st.chat_input("審判役に指示や追加質問を入力...")

        if user_query:
            # ユーザーの質問を表示＆保存
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # AI（審判役）の回答生成
            with st.chat_message("assistant"):
                with st.spinner("🤖 審判役が追加質問を分析し成果物を再構築中..."):
                    sys_chat_judge = (
                        "あなたは最高権威の審判役兼ファシリテーターです。\n"
                        "ユーザーから提出された追加質問や指示に対し、これまでの議論と直前の成果物を踏まえて回答・成果物の再構築を行ってください。\n\n"
                        "【絶対遵守ルール】\n"
                        "1. ユーザーが変更や修正を求めている場合、変更点の説明だけでなく、**必ず途中省略なしの完全な改訂成果物**を提示してください。"
                    )

                    # チャット履歴の構築
                    history_str = ""
                    for h in st.session_state.chat_history[:-1]:
                        role_name = "ユーザー" if h["role"] == "user" else "審判役"
                        history_str += f"\n【{role_name}】\n{h['content']}\n"

                    prompt_contents = []
                    if st.session_state.saved_image_bytes:
                        prompt_contents.append(Image.open(io.BytesIO(st.session_state.saved_image_bytes)))

                    prompt_contents.append(
                        f"### 当初の目的・テーマ\n{st.session_state.topic_used}\n\n"
                        f"### 審判役の初期結論・成果物\n{st.session_state.discussion_result['judge']}\n\n"
                        f"### 過去の追加質問履歴\n{history_str if history_str else 'なし'}\n\n"
                        f"### 今回のユーザーからの追加指示・質問\n{user_query}"
                    )

                    try:
                        ans_text = call_gemini_with_jitter_retry(prompt_contents, sys_chat_judge, 0.2, use_search=False)
                        st.markdown(ans_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans_text})
                    except Exception as e:
                        st.error(f"追加質問への回答中にエラーが発生しました: {e}")
