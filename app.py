from datetime import datetime
import os
import time
import re
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
import streamlit as st

# --------------------------------------------------
# ページ初期設定
# --------------------------------------------------
st.set_page_config(
    page_title="AIマルチエージェント討論 (検索グラウンディング搭載版)",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 AIマルチエージェント討論システム (最新情報対応版)")
st.caption(
    "Gemini 3.6 Flash + Google検索機能による 💡提案役 vs ⚡批判役 vs ↩️反論 vs 🏆審判役 の最新ディベート"
)

# --------------------------------------------------
# APIキーの取得
# --------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ 設定・操作")
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key を入力してください:", type="password"
        )
        st.caption("[Google AI Studio](https://aistudio.google.com/) で取得可能です。")

    if st.button("🗑️ 画面と履歴をリセット", use_container_width=True):
        st.session_state.discussion_result = None
        st.rerun()

if not api_key:
    st.info("👈 サイドバーから Gemini API キーを設定してください。")
    st.stop()

# SDK初期化
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

# --------------------------------------------------
# ユーティリティ関数
# --------------------------------------------------
def compress_image(image: Image.Image, max_size=(600, 600)) -> Image.Image:
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def sanitize_input(text: str) -> str:
    return text.replace("```", "'''").strip()

def call_gemini_with_smart_retry(
    contents, system_instruction: str, temperature: float, status_container, max_retries: int = 4
):
    """Google検索ツール（グラウンディング）と自動リトライを統合したAPIコール関数"""
    
    # Google検索をツールとして定義（最新情報の取得に必須）
    search_tool = types.Tool(google_search=types.GoogleSearch())
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=2000,
        tools=[search_tool],
    )
    
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
        except APIError as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 20
                match = re.search(r"retry in ([\d\.]+)s", error_msg)
                if match:
                    wait_time = int(float(match.group(1))) + 2

                status_container.warning(f"⚠️ 無料枠のAPI制限に到達しました。{wait_time}秒待機して自動再開します（{attempt+1}回目のリトライ）...")
                time.sleep(wait_time)
                status_container.info("🔄 処理を再開します...")
            else:
                raise e

def manual_wait(seconds: int, message: str, status_container):
    status_container.write(f"⏳ {message} (API制限回避のため {seconds}秒 待機中...)")
    time.sleep(seconds)

# --------------------------------------------------
# 入力フォーム領域
# --------------------------------------------------
topic_raw = st.text_area(
    "検討したいテーマ、アイデア、最新トレンドなどを入力してください:",
    placeholder="例：2026年現在の最新AIトレンドを踏まえて、新規事業のアイデアを議論してください。",
    height=120,
)

uploaded_file = st.file_uploader(
    "📷 画像・資料・デザイン案（任意）",
    type=["png", "jpg", "jpeg", "webp"],
)

processed_image = None
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    processed_image = compress_image(raw_image)
    st.image(processed_image, caption="添付画像（最適化済み）", width=250)

if "discussion_result" not in st.session_state:
    st.session_state.discussion_result = None

# --------------------------------------------------
# 議論実行処理
# --------------------------------------------------
if st.button("🚀 最新情報を踏まえて討論を開始する", type="primary", use_container_width=True):
    topic = sanitize_input(topic_raw)

    if not topic and processed_image is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        try:
            with st.status(
                "🤖 AIエージェントたちが最新情報を検索しながら議論を構築中...",
                expanded=True,
            ) as status:
                
                BASE_WAIT = 15

                # --- PHASE 1: 提案役 ---
                status.write("1/4 💡 【提案役】最新情報を調査しつつ、強みと推進案を構築中...")
                sys_proposer = (
                    "あなたはこのアイデアを成功に導く提案役です。"
                    "Google検索を活用し、最新データに基づいた論理的な主張を展開してください。\n\n"
                    "【重要ルール】\n"
                    "・挨拶、前置き、自己紹介などの無駄な言葉は一切排除してください。\n"
                    "・文脈や重要なデータ、検索結果のファクトは絶対に省略しないでください。\n"
                    "・以下の構造で情報の密度を高く出力してください。\n"
                    "1. [コアとなる主張]\n"
                    "2. [根拠となる最新データ/ファクト]\n"
                    "3. [具体的な推進アプローチ]"
                )
                
                contents_p1 = [processed_image, f"### 検討テーマ\n{topic}"] if processed_image else [f"### 検討テーマ\n{topic}"]
                res_p1 = call_gemini_with_smart_retry(
                    contents=contents_p1, system_instruction=sys_proposer, temperature=0.7, status_container=status
                )
                proposer_text = res_p1.text
                
                manual_wait(BASE_WAIT, "提案役の意見をまとめています", status)

                # --- PHASE 2: 批判役 ---
                status.write("2/4 ⚡ 【批判役】最新の競合状況やリスクを検証中...")
                sys_critic = (
                    "あなたは鋭い視点を持つ批判役（リスクアナリスト）です。"
                    "提案役の意見を踏まえ、Google検索で最新のリスク要因、競合状況、技術的・法的な課題を洗い出してください。\n\n"
                    "【重要ルール】\n"
                    "・挨拶や感情的な表現は避け、冷徹かつ論理的に反証してください。\n"
                    "・以下の構造で出力してください。\n"
                    "1. [提案の致命的な弱点・リスク]\n"
                    "2. [それを裏付ける最新の競合・市場データ]\n"
                    "3. [解決が困難なボトルネック]"
                )
                
                contents_p2 = [f"### 検討テーマ\n{topic}\n\n### 提案役の意見\n{proposer_text}"]
                res_p2 = call_gemini_with_smart_retry(
                    contents=contents_p2, system_instruction=sys_critic, temperature=0.7, status_container=status
                )
                critic_text = res_p2.text
                
                manual_wait(BASE_WAIT, "批判役の意見をまとめています", status)

                # --- PHASE 3: 反論・改善（提案役） ---
                status.write("3/4 ↩️ 【反論役】批判に対する代替案や解決策を模索中...")
                sys_rebutter = (
                    "あなたは提案役のサポートに入り、批判役の指摘を乗り越える解決策を提示する反論役です。"
                    "批判役の指摘事項をGoogle検索で深掘りし、それを克服する具体的な最新のアプローチを提案してください。\n\n"
                    "【重要ルール】\n"
                    "1. [批判の受容とピボット案]\n"
                    "2. [リスクを軽減する最新技術/事例]\n"
                    "3. [実現可能性を再定義する次のステップ]"
                )
                
                contents_p3 = [
                    f"### テーマ\n{topic}\n\n"
                    f"### 提案役\n{proposer_text}\n\n"
                    f"### 批判役からの指摘\n{critic_text}"
                ]
                res_p3 = call_gemini_with_smart_retry(
                    contents=contents_p3, system_instruction=sys_rebutter, temperature=0.7, status_container=status
                )
                rebutter_text = res_p3.text
                
                manual_wait(BASE_WAIT, "反論役の意見をまとめています", status)

                # --- PHASE 4: 審判役（サマリー） ---
                status.write("4/4 🏆 【審判役】議論全体を俯瞰し、最終的な結論とネクストアクションを生成中...")
                sys_judge = (
                    "あなたはこれまでの議論を客観的に評価し、最終的な結論を下す審判役です。"
                    "提案、批判、反論のすべてを統合し、実用的な結論を出してください。\n\n"
                    "【重要ルール】\n"
                    "・箇条書きを活用し、視覚的にわかりやすく出力してください。\n"
                    "1. [議論のサマリー（争点）]\n"
                    "2. [総合評価（Go or No-Go、あるいは条件付きGo）]\n"
                    "3. [直近1ヶ月で取るべき具体的なアクションプラン]"
                )
                
                contents_p4 = [
                    f"### テーマ\n{topic}\n\n"
                    f"### 提案\n{proposer_text}\n\n"
                    f"### 批判\n{critic_text}\n\n"
                    f"### 反論・解決策\n{rebutter_text}"
                ]
                res_p4 = call_gemini_with_smart_retry(
                    contents=contents_p4, system_instruction=sys_judge, temperature=0.4, status_container=status
                )
                judge_text = res_p4.text

                # セッションステートに保存
                st.session_state.discussion_result = {
                    "proposer": proposer_text,
                    "critic": critic_text,
                    "rebutter": rebutter_text,
                    "judge": judge_text
                }
                
                status.update(label="✅ 討論が完了しました！", state="complete", expanded=False)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# --------------------------------------------------
# 結果表示
# --------------------------------------------------
if st.session_state.discussion_result:
    res = st.session_state.discussion_result
    
    st.markdown("---")
    st.header("🗣️ 議論の結果")
    
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

