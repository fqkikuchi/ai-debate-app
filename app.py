from datetime import datetime
import os
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
    page_title="AIマルチエージェント討論 (最新情報・グラウンディング対応)",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AIマルチエージェント討論システム (Gemini 3.6 Flash対応版)")
st.caption(
    "💡提案役 vs ⚡批判役 vs ↩️反論 vs 🏆審判役 による最新Web検索連携ディベート"
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

    # 使用モデルの設定
    selected_model = st.selectbox(
        "使用モデル",
        ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash", "カスタム入力"],
        index=0,
    )
    
    if selected_model == "カスタム入力":
        selected_model = st.text_input("モデル名を入力:", value="gemini-3.6-flash")

    if st.button("🗑️ 画面と履歴をリセット", use_container_width=True):
        st.session_state.discussion_result = None
        st.rerun()

if not api_key:
    st.info("👈 サイドバーから Gemini API キーを設定してください。")
    st.stop()

# GenAI クライアントの初期化
client = genai.Client(api_key=api_key)

# --------------------------------------------------
# 3. ユーティリティ関数
# --------------------------------------------------
def compress_image(image: Image.Image, max_size=(600, 600)) -> Image.Image:
    """画像の圧縮（API送信サイズ最適化）"""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def sanitize_input(text: str) -> str:
    """入力の安全性向上"""
    return text.replace("```", "'''").strip()

def call_gemini_with_smart_retry(
    contents, system_instruction: str, temperature: float, status_container, max_retries: int = 4
):
    """Google検索ツール（グラウンディング）と自動リトライを統合したAPIコール関数"""
    
    current_date = datetime.now().strftime("%Y年%m月%d日")
    context_header = (
        f"【最重要前提】本日は {current_date} です。"
        f"現在、Geminiの最新モデル（Gemini 3.6 Flash等）や最新の環境が利用可能です。"
        f"過去の知識に縛られず、常に最新の事実に基づいて思考・検索してください。\n\n"
    )
    full_system_instruction = context_header + system_instruction

    search_tool = types.Tool(google_search=types.GoogleSearch())
    
    config = types.GenerateContentConfig(
        system_instruction=full_system_instruction,
        temperature=temperature,
        max_output_tokens=8192,  # 💡ここでトークン上限を最大値に解放しています
        tools=[search_tool],
    )
    
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=selected_model, contents=contents, config=config
            )
        except APIError as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt == max_retries - 1:
                    raise e
                wait_time = 20
                status_container.warning(f"⚠️ API制限に到達しました。{wait_time}秒待機して自動再開します（{attempt+1}/{max_retries}）...")
                time.sleep(wait_time)
                status_container.info("🔄 処理を再開します...")
            else:
                raise e

def manual_wait(seconds: int, message: str, status_container):
    status_container.write(f"⏳ {message} (制限回避のため {seconds}秒 待機中...)")
    time.sleep(seconds)

# --------------------------------------------------
# 4. メインUI (入力領域)
# --------------------------------------------------
topic_raw = st.text_area(
    "検討したいテーマ、アイデア、最新トレンドなどを入力してください:",
    placeholder="例：2026年現在のAI技術を踏まえた最新の作業自動化アプローチについて議論してください。",
    height=100,
)

uploaded_file = st.file_uploader(
    "📷 関連画像・資料（任意）",
    type=["png", "jpg", "jpeg", "webp"],
)

processed_image = None
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    processed_image = compress_image(raw_image)
    st.image(processed_image, caption="添付画像（最適化済み）", width=250)

if "discussion_result" not in st.session_state:
    st.session_state.discussion_result = None
    st.session_state.topic_used = ""

# --------------------------------------------------
# 5. 討論実行ロジック
# --------------------------------------------------
if st.button("🚀 最新情報を検索して討論を開始", type="primary", use_container_width=True):
    topic = sanitize_input(topic_raw)

    if not topic and processed_image is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        try:
            with st.status(
                "🤖 AIエージェントたちが最新情報を検証しながら議論中...",
                expanded=True,
            ) as status:
                
                BASE_WAIT = 10

                # --- PHASE 1: 提案役 ---
                status.write("1/4 💡 【提案役】Web検索で最新事実を調査し、主張を構築中...")
                sys_proposer = (
                    "あなたはこのアイデアを成功に導く提案役です。"
                    "Google検索を活用し、最新データに基づいた論理的な主張を展開してください。\n"
                    "・前置きや挨拶は排除し、1. [コアとなる主張], 2. [最新根拠/ファクト], 3. [推進アプローチ] で記述してください。"
                )
                
                contents_p1 = [processed_image, f"### 検討テーマ\n{topic}"] if processed_image else [f"### 検討テーマ\n{topic}"]
                res_p1 = call_gemini_with_smart_retry(contents_p1, sys_proposer, 0.7, status)
                proposer_text = res_p1.text
                
                manual_wait(BASE_WAIT, "提案役の意見を整理中", status)

                # --- PHASE 2: 批判役 ---
                status.write("2/4 ⚡ 【批判役】競合状況や潜在リスクを検証中...")
                sys_critic = (
                    "あなたは鋭い視点を持つ批判役です。"
                    "提案役の意見を踏まえ、Google検索で最新のリスク要因や競合課題を洗い出してください。\n"
                    "・1. [提案の弱点・リスク], 2. [裏付ける最新市場/競合データ], 3. [解決困難なボトルネック] で記述してください。"
                )
                
                contents_p2 = [f"### 検討テーマ\n{topic}\n\n### 提案役の主張\n{proposer_text}"]
                res_p2 = call_gemini_with_smart_retry(contents_p2, sys_critic, 0.7, status)
                critic_text = res_p2.text
                
                manual_wait(BASE_WAIT, "批判役の意見を整理中", status)

                # --- PHASE 3: 反論役 ---
                status.write("3/4 ↩️ 【反論役】批判を乗り越える最新の代替案を策定中...")
                sys_rebutter = (
                    "あなたは提案役をサポートし、批判を克服する反論役です。"
                    "批判役の指摘事項をGoogle検索で深掘りし、それを乗り越える最新アプローチを提示してください。\n"
                    "・1. [批判の受容とピボット案], 2. [リスクを軽減する最新技術/事例], 3. [実現に向けた次のステップ] で記述してください。"
                )
                
                contents_p3 = [
                    f"### テーマ\n{topic}\n\n"
                    f"### 提案役\n{proposer_text}\n\n"
                    f"### 批判役の指摘\n{critic_text}"
                ]
                res_p3 = call_gemini_with_smart_retry(contents_p3, sys_rebutter, 0.7, status)
                rebutter_text = res_p3.text
                
                manual_wait(BASE_WAIT, "反論役の意見を整理中", status)

                # --- PHASE 4: 審判役 ---
                status.write("4/4 🏆 【審判役】議論全体を総括し、最終結論を生成中...")
                sys_judge = (
                    "あなたは客観的な審判役です。すべての議論を評価し、実用的な結論を下してください。\n"
                    "・1. [議論のサマリー], 2. [総合評価 (Go / No-Go / 条件付きGo)], 3. [具体的なアクションプラン] で記述してください。"
                )
                
                contents_p4 = [
                    f"### テーマ\n{topic}\n\n"
                    f"### 提案\n{proposer_text}\n\n"
                    f"### 批判\n{critic_text}\n\n"
                    f"### 反論・解決策\n{rebutter_text}"
                ]
                res_p4 = call_gemini_with_smart_retry(contents_p4, sys_judge, 0.4, status)
                judge_text = res_p4.text

                # セッションへ保存
                st.session_state.topic_used = topic
                st.session_state.discussion_result = {
                    "proposer": proposer_text,
                    "critic": critic_text,
                    "rebutter": rebutter_text,
                    "judge": judge_text,
                }
                
                status.update(label="✅ 討論が正常に完了しました！", state="complete", expanded=False)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# --------------------------------------------------
# 6. 結果の描画 UI と 審判役の続き生成機能
# --------------------------------------------------
if st.session_state.discussion_result:
    res = st.session_state.discussion_result
    
    st.markdown("---")
    st.header("🗣️ 議論結果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("💡 提案役の主張")
            st.write(res["proposer"])
            
        with st.container(border=True):
            st.subheader("↩️ 反論・解決案 (提案役側)")
            st.write(res["rebutter"])
            
    with col2:
        with st.container(border=True):
            st.subheader("⚡ 批判役の指摘")
            st.write(res["critic"])
            
    st.markdown("---")
    st.header("🏆 審判役の最終結論")
    with st.container(border=True):
        st.write(res["judge"])

    # 💡 審判役の出力が途切れた場合の追加生成ボタン
    st.markdown("---")
    if st.button("📝 審判役の結論が途切れている場合、続きを生成する"):
        try:
            with st.status("🔄 審判役の続きを生成しています...", expanded=True) as status:
                
                # 直前の文章の末尾数文字を取得して文脈として渡す
                last_text = res["judge"][-300:] if len(res["judge"]) > 300 else res["judge"]
                
                sys_judge_continue = (
                    "あなたは客観的な審判役です。先ほどの出力がトークン制限により途中で切れてしまいました。\n"
                    "以下の【出力済みのテキスト】の続きから、残りの結論を出力してください。\n"
                    "※挨拶や重複する内容は書かず、完全に続きの文章から書き始めてください。\n\n"
                    f"【出力済みのテキスト】\n{last_text}"
                )
                
                contents_cont = [
                    f"### テーマ\n{st.session_state.topic_used}\n\n"
                    f"### 提案\n{res['proposer']}\n\n"
                    f"### 批判\n{res['critic']}\n\n"
                    f"### 反論・解決策\n{res['rebutter']}"
                ]
                
                res_cont = call_gemini_with_smart_retry(contents_cont, sys_judge_continue, 0.4, status)
                
                # 続きのテキストを結合して再描画
                st.session_state.discussion_result["judge"] += "\n\n" + res_cont.text
                status.update(label="✅ 続きの生成が完了しました！", state="complete")
                
            st.rerun()
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
