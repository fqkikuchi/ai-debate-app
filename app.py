import os
import io
from google import genai
from google.genai.errors import APIError
import streamlit as st
from PIL import Image

# --------------------------------------------------
# ページ初期設定
# --------------------------------------------------
st.set_page_config(
    page_title="AIマルチエージェント討論 (堅牢版)",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AIマルチエージェント討論システム")
st.caption("Gemini 2.0 Flashによる 提案役 vs 批判役 vs 審判役 の徹底議論")

# APIキーの取得
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("APIキーが設定されていません。環境変数またはStreamlit Secretsに GEMINI_API_KEY を設定してください。")
    st.stop()

# SDK初期化
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

# 画像リサイズ関数（トークン節約と高速化のため）
def compress_image(image: Image.Image, max_size=(800, 800)) -> Image.Image:
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

# --------------------------------------------------
# 入力フォーム領域
# --------------------------------------------------
topic = st.text_area(
    "検討したいテーマ、アイデア、文章の下書きなどを入力してください:",
    placeholder="例：この新商品のチラシデザイン案について、ターゲット層への訴求力を議論してください。",
    height=150,
)

uploaded_file = st.file_uploader(
    "📷 画像・資料・デザイン案（任意・自動最適化されます）", 
    type=["png", "jpg", "jpeg", "webp"]
)

processed_image = None
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    processed_image = compress_image(raw_image) # 軽量化処理
    st.image(processed_image, caption="添付画像（最適化済み）", width=300)

if "discussion_result" not in st.session_state:
    st.session_state.discussion_result = None

# --------------------------------------------------
# 議論実行処理
# --------------------------------------------------
if st.button("🚀 議論を開始する", type="primary"):
    if not topic.strip() and processed_image is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        try:
            with st.status("🤖 AIエージェントたちが議論を進行中...", expanded=True) as status:

                # 画像がある場合はコンテンツリストに追加
                def get_contents(prompt_text):
                    if processed_image:
                        return [prompt_text, processed_image]
                    return [prompt_text]

                # --- 1. 提案役（肯定派） ---
                st.write("1/3 💡 提案役がポジティブな視点で分析中...")
                prompt_proposer = f"""
                テーマ/入力文: 「{topic}」
                あなたはこのアイデアや文章、デザインを成功させたい「熱心な提案役（肯定派）」です。
                この内容の強み、具体的で現実的な実践方法、期待できる成果を論理的かつ情熱的に主張してください。
                （※画像がある場合は視覚的な強みも評価してください）
                """
                res_proposer = client.models.generate_content(
                    model=MODEL_NAME, contents=get_contents(prompt_proposer)
                )
                proposer_text = res_proposer.text

                # --- 2. 批判役（悪魔の代弁者） ---
                st.write("2/3 ⚡ 批判役がリスクや盲点を徹底ツッコミ中...")
                prompt_critic = f"""
                テーマ/入力文: 「{topic}」
                【提案役の主張】:
                {proposer_text}

                あなたは徹底的な「批判役（悪魔の代弁者）」です。
                提案役の甘い見通し、見落としているリスク、懸念点、画像上の欠点を論理的かつ容赦なく批判してください。
                """
                # 2回目以降はトークン節約のため、画像は含めずテキストのみで議論を深める（必要に応じて画像を含めることも可）
                res_critic = client.models.generate_content(
                    model=MODEL_NAME, contents=prompt_critic
                )
                critic_text = res_critic.text

                # --- 3. 審判役（最終結論） ---
                st.write("3/3 🏆 審判役が両者の意見を統合中...")
                prompt_judge = f"""
                テーマ/入力文: 「{topic}」
                【提案役の主張】: {proposer_text}
                【批判役の指摘】: {critic_text}

                あなたは優れた洞察力を持つ「審判（まとめ役）」です。
                1. **【分析】提案の評価点 と 批判の重く受け止めるべき指摘**
                2. **【改善の方向性】リスクを回避するためのポイント**
                3. **【最終完成版・アクションプラン】** (ビジネスなら実行プラン、文章ならそのまま使える完成文)
                をわかりやすく出力してください。
                """
                res_judge = client.models.generate_content(
                    model=MODEL_NAME, contents=prompt_judge
                )
                judge_text = res_judge.text

                status.update(label="✅ 議論が完了しました！", state="complete", expanded=False)

            # セッション保存
            st.session_state.discussion_result = {
                "topic": topic,
                "proposer": proposer_text,
                "critic": critic_text,
                "judge": judge_text,
            }

        except APIError as e:
            st.error(f"APIエラーが発生しました。時間を置いて再試行してください: {e}")
        except Exception as e:
            st.error(f"予期せぬエラーが発生しました: {e}")

# --------------------------------------------------
# 結果表示領域
# --------------------------------------------------
if st.session_state.discussion_result:
    res = st.session_state.discussion_result

    st.divider()
    st.subheader("📊 議論結果")

    tab1, tab2, tab3 = st.tabs([
        "🏆 最終結論・完成版",
        "💡 1. 提案役（肯定派）",
        "⚡ 2. 批判役（悪魔の代弁者）"
    ])

    with tab1:
        st.success(res["judge"])
        full_markdown = (
            f"# 検討テーマ:\n{res['topic']}\n\n"
            f"---\n## 🏆 最終結論・完成版\n{res['judge']}\n\n"
            f"---\n## 💡 提案役の主張\n{res['proposer']}\n\n"
            f"---\n## ⚡ 批判役の指摘\n{res['critic']}"
        )
        st.download_button(
            label="📝 議論結果をダウンロード (.md)",
            data=full_markdown,
            file_name="discussion_result.md",
            mime="text/markdown",
        )

    with tab2:
        st.info(res["proposer"])

    with tab3:
        st.warning(res["critic"])
