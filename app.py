from datetime import datetime
import os
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
import streamlit as st

# --------------------------------------------------
# ページ初期設定
# --------------------------------------------------
st.set_page_config(
    page_title="AIマルチエージェント討論 (ハイブリッド構成版)",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 AIマルチエージェント討論システム (ハイブリッド構成版)")
st.caption(
    "Gemini 3.5 Flash-Lite & 3.6 Flash による 💡提案役 vs ⚡批判役 vs ↩️反論 vs 🏆審判役 の4段階ディベート"
)

# --------------------------------------------------
# APIキーの取得（Secrets / 環境変数 / サイドバー）
# --------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

with st.sidebar:
    st.header("⚙️ 設定・操作")
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key を入力してください:", type="password"
        )
        st.caption(
            "[Google AI Studio](https://aistudio.google.com/) で取得可能です。"
        )

    if st.button("🗑️ 画面と履歴をリセット", use_container_width=True):
        st.session_state.discussion_result = None
        st.rerun()

if not api_key:
    st.info("👈 サイドバーから Gemini API キーを設定してください。")
    st.stop()

# SDK初期化
client = genai.Client(api_key=api_key)

# --------------------------------------------------
# モデルの定義（適材適所で使い分ける）
# --------------------------------------------------
# 処理が重い役割（画像解析、単純な反論）用：無料枠に優しい軽量版
MODEL_LITE = "gemini-3.5-flash-lite"
# 論理的思考が求められる役割（批判、最終結論）用：高品質版
MODEL_PRO = "gemini-3.6-flash"

# --------------------------------------------------
# ユーティリティ関数（無料枠対策・セキュリティ）
# --------------------------------------------------
def compress_image(image: Image.Image, max_size=(500, 500)) -> Image.Image:
    """画像の自動リサイズ（トークン節約のため小さめに設定）"""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def sanitize_input(text: str) -> str:
    """簡易インジェクション対策"""
    return text.replace("```", "'''").strip()

def call_gemini_with_retry(
    contents, system_instruction: str, temperature: float, model_name: str, max_retries: int = 3
):
    """APIコールとエラーハンドリング（モデルを動的に切り替え）"""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=600, # 無料枠対策: 出力トークンを絞る
    )
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
        except APIError as e:
            if attempt == max_retries - 1:
                raise e
            # 429 (Rate Limit) エラー対策: リトライ待機時間を長くする
            time.sleep(15 * (attempt + 1))

def wait_for_api_limit(seconds: int, message: str):
    """APIのRate Limit回避のためのカウントダウンUI"""
    placeholder = st.empty()
    for i in range(seconds, 0, -1):
        placeholder.info(f"⏳ {message}... API制限回避のため待機中 ({i}秒)")
        time.sleep(1)
    placeholder.empty()

# --------------------------------------------------
