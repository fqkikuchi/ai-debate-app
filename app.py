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
# APIキーの取得
# --------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

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

# --------------------------------------------------
# モデルの定義（ハイブリッド構成用に2種類用意）
# --------------------------------------------------
# 処理が重い役割（画像解析、単純な反論）用：無料枠に優しい軽量版
MODEL_LITE = "gemini-3.5-flash-lite"
# 論理的思考が求められる役割（批判、最終結論）用：高品質版
MODEL_PRO = "gemini-3.6-flash"

# --------------------------------------------------
# ユーティリティ関数
# --------------------------------------------------
def compress_image(image: Image.Image, max_size=(500, 500)) -> Image.Image:
    """画像の自動リサイズ（トークン節約のためさらに小さく設定）"""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def sanitize_input(text: str) -> str:
    return text.replace("```", "'''").strip()

def call_gemini_with_retry(
    contents, system_instruction: str, temperature: float, model_name: str, max_retries: int = 3
):
    """APIコールとエラーハンドリング（引数に model_name を追加）"""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=600, # 無料枠対策: 出力トークンをさらに絞る
    )
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
        except APIError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(15 * (attempt + 1))

def wait_for_api_limit(seconds: int, message: str):
    """APIのRate Limit回避のためのカウントダウンUI"""
    placeholder = st.empty()
    for i in range(seconds, 0, -1):
        placeholder.info(f"⏳ {message}... API制限回避のため待機中 ({i}秒)")
        time.sleep(1)
    placeholder.empty()

# --------------------------------------------------
# 入力フォーム領域
# --------------------------------------------------
topic_raw = st.text_area(
    "検討したいテーマ、アイデア、文章の下書きなどを入力してください:",
    placeholder="例：新商品のチラシデザイン案について、ターゲット層への訴求力を議論してください。",
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
# 議論実行処理 (4フェーズ・ディベート)
# --------------------------------------------------
if st.button("🚀 討論を開始する", type="primary", use_container_width=True):
    topic = sanitize_input(topic_raw)

    if not topic and processed_image is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        try:
            with st.status(
                "🤖 AIエージェントたちがディベートを実行中... (約1分少々かかります)",
                expanded=True,
            ) as status:
                
                WAIT_SECONDS = 15 # 各APIコール間の待機時間

                # --- PHASE 1: 提案役（Lite版） ---
                st.write("1/4 💡 【提案役】強みと推進案を構築中...")
                sys_proposer = (
                    "あなたはこのアイデアや企画を成功させたい「熱心な提案役」です。"
                    "画像が添付されている場合は、その視覚的特徴も言語化して取り入れ、"
                    "強みや期待できる成果を論理的に主張してください。"
                )
                
                # 画像がある場合、Phase 1のみに画像を渡す
                contents_p1 = [processed_image, f"### 検討テーマ\n{topic}"] if processed_image else [f"### 検討テーマ\n{topic}"]
                
                res_p1 = call_gemini_with_retry(
                    contents=contents_p1, system_instruction=sys_proposer, temperature=0.7, model_name=MODEL_LITE
                )
                proposer_text = res_p1.text
                
                wait_for_api_limit(WAIT_SECONDS, "提案役の意見をまとめています")

                # --- PHASE 2: 批判役（Pro版） ---
                st.write("2/4 ⚡ 【批判役】リスクと欠点を徹底検証中...")
                sys_critic = (
                    "あなたは徹底的な「批判役（悪魔の代弁者）」です。"
                    "提案役の主張に対し、見落としているリスクや構造的欠点を冷静に3点以内で指摘してください。"
                )
                # 画像は送らない（トークン大幅節約）
                prompt_c1 = f"### 検討テーマ\n{topic}\n\n### 提案役の主張\n{proposer_text}"
                res_c1 = call_gemini_with_retry(
                    contents=[prompt_c1], system_instruction=sys_critic, temperature=0.2, model_name=MODEL_PRO
                )
                critic_text = res_c1.text

                wait_for_api_limit(WAIT_SECONDS, "批判役の指摘を精査しています")

                # --- PHASE 3: 提案役の反論（Lite版） ---
                st.write("3/4 ↩️ 【提案役】批判に対する誤解の解明・対案（反論）を作成中...")
                sys_rebuttal = (
                    "あなたは提案役です。批判役の指摘を受け止めつつ、"
                    "批判を吸収した「現実的な補足・対案」を簡潔に提示してください。"
                )
                prompt_r1 = f"### 批判役の指摘\n{critic_text}"
                res_r1 = call_gemini_with_retry(
                    contents=[prompt_r1], system_instruction=sys_rebuttal, temperature=0.5, model_name=MODEL_LITE
                )
                rebuttal_text = res_r1.text

                wait_for_api_limit(WAIT_SECONDS, "最終判断の準備をしています")

                # --- PHASE 4: 審判役（Pro版） ---
                st.write("4/4 🏆 【審判役】すべての議論を統合し、最適解を策定中...")
                sys_judge = (
                    "あなたは公正な「審判（まとめ役）」です。"
                    "提案、批判、反論を精査し、リスクを最小化しつつ効果を最大化する【最終完成版】を作成してください。"
                )
                prompt_j1 = (
                    f"### 検討テーマ\n{topic}\n\n"
                    f"### 1. 提案\n{proposer_text}\n"
                    f"### 2. 批判\n{critic_text}\n"
                    f"### 3. 反論\n{rebuttal_text}\n\n"
                    "出力構成：\n1. **【総合分析】**\n2. **【回避すべきリスク】**\n3. **【最終完成版・アクションプラン】**"
                )
                res_j1 = call_gemini_with_retry(
                    contents=[prompt_j1], system_instruction=sys_judge, temperature=0.3, model_name=MODEL_PRO
                )
                judge_text = res_j1.text

                status.update(
                    label="✅ 討論完了！最適化された結論が生成されました。",
                    state="complete",
                    expanded=False,
                )

            # セッション保存
            st.session_state.discussion_result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "topic": topic,
                "proposer": proposer_text,
                "critic": critic_text,
                "rebuttal": rebuttal_text,
                "judge": judge_text,
            }

        except APIError as e:
            st.error(f"🚨 APIエラーが発生しました。時間を置いて再度お試しください: {e}")
        except Exception as e:
            st.error(f"🚨 予期せぬエラーが発生しました: {e}")

# --------------------------------------------------
# 結果表示領域 (変更なし)
# --------------------------------------------------
if st.session_state.discussion_result:
    res = st.session_state.discussion_result
    st.divider()
    st.subheader("📊 議論結果および最終完成版")
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 最終結論", "💡 提案", "⚡ 批判", "↩️ 反論"])
    with tab1:
        st.success(res["judge"])
    with tab2:
        st.info(res["proposer"])
    with tab3:
        st.warning(res["critic"])
    with tab4:
        st.markdown(res["rebuttal"])
