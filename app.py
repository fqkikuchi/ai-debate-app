from datetime import datetime
import io
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
    page_title="AIマルチエージェント討論 (プロダクト版)",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 AIマルチエージェント討論システム (プロダクト版)")
st.caption(
    "Gemini 3.6 Flash による 💡提案役 vs ⚡批判役 vs ↩️反論 vs 🏆審判役 の4段階ディベート"
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
# 無料枠で安定して動く現行モデルを指定
MODEL_NAME = "gemini-3.6-flash"


# --------------------------------------------------
# ユーティリティ関数（無料枠対策・セキュリティ）
# --------------------------------------------------
def compress_image(image: Image.Image, max_size=(600, 600)) -> Image.Image:
    """画像の自動リサイズ（無料枠の入力トークンを節約するために少し小さめに設定）"""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def sanitize_input(text: str) -> str:
    """簡易インジェクション対策"""
    return text.replace("```", "'''").strip()


def call_gemini_with_retry(
    contents, system_instruction: str, temperature: float, max_retries: int = 3
):
    """APIエラー時のリトライ処理（無料枠のRate Limit対策を強化）"""
    # 無料枠のTPM（1分間のトークン上限）対策として最大出力トークンを制限
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=800, # 長すぎる回答を防止
    )
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
        except APIError as e:
            if attempt == max_retries - 1:
                raise e
            # 429 (Rate Limit) エラー対策: 待機時間を長くする (10秒, 20秒...)
            wait_time = 10 * (attempt + 1)
            time.sleep(wait_time)


# --------------------------------------------------
# 入力フォーム領域
# --------------------------------------------------
topic_raw = st.text_area(
    "検討したいテーマ、アイデア、文章の下書きなどを入力してください:",
    placeholder="例：新商品のチラシデザイン案について、ターゲット層への訴求力を議論してください。",
    height=120,
)

uploaded_file = st.file_uploader(
    "📷 画像・資料・デザイン案（任意・自動最適化）",
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
                "🤖 AIエージェントたちがディベートを実行中...（無料枠APIへの負荷を抑えながら進行します）",
                expanded=True,
            ) as status:

                # 画像添付ヘルパー
                def build_contents(text_data):
                    return (
                        [processed_image, text_data]
                        if processed_image
                        else [text_data]
                    )
                
                # 無料枠RPM（リクエスト数上限）対策のウェイト
                DELAY_BETWEEN_CALLS = 8

                # --- PHASE 1: 提案役（肯定派） ---
                st.write("1/4 💡 【提案役】強みと推進案を構築中...")
                sys_proposer = (
                    "あなたはこのアイデアや企画を成功させたい「熱心な提案役（肯定派）」です。"
                    "強み、具体的な実践方法、期待できる成果を要点を絞って論理的に主張してください。"
                )
                prompt_p1 = f"### 検討テーマ\n{topic}"
                res_p1 = call_gemini_with_retry(
                    build_contents(prompt_p1), sys_proposer, temperature=0.7
                )
                proposer_text = res_p1.text
                
                time.sleep(DELAY_BETWEEN_CALLS) # インターバル

                # --- PHASE 2: 批判役（悪魔の代弁者） ---
                st.write("2/4 ⚡ 【批判役】リスクと欠点を徹底検証中...")
                sys_critic = (
                    "あなたは徹底的な「批判役（悪魔の代弁者）」です。"
                    "提案役の見落としているリスク、甘い見通し、視覚的・構造的欠点を冷静に指摘してください。長文を避け、3〜4点に絞ってください。"
                )
                prompt_c1 = f"### 検討テーマ\n{topic}\n\n### 提案役の主張\n{proposer_text}"
                res_c1 = call_gemini_with_retry(
                    build_contents(prompt_c1), sys_critic, temperature=0.2
                )
                critic_text = res_c1.text

                time.sleep(DELAY_BETWEEN_CALLS) # インターバル

                # --- PHASE 3: 提案役の反論（リバッタル） ---
                st.write("3/4 ↩️ 【提案役】批判に対する誤解の解明・対案（反論）を作成中...")
                sys_rebuttal = (
                    "あなたは提案役です。批判役からの指摘を受け止めつつ、"
                    "不当な懸念への反論、または批判を吸収した「現実的な補足・対案」を簡潔に提示してください。"
                )
                prompt_r1 = f"### 批判役の指摘\n{critic_text}"
                res_r1 = call_gemini_with_retry(
                    prompt_r1, sys_rebuttal, temperature=0.5
                )
                rebuttal_text = res_r1.text

                time.sleep(DELAY_BETWEEN_CALLS) # インターバル

                # --- PHASE 4: 審判役（最終統合・アクションプラン） ---
                st.write("4/4 🏆 【審判役】すべての議論を統合し、最適解を策定中...")
                sys_judge = (
                    "あなたは公正な「審判（まとめ役）」です。"
                    "提案、批判、反論のプロセスを精査し、リスクを最小化しつつ効果を最大化する【最終完成版】を作成してください。"
                )
                prompt_j1 = (
                    f"### 検討テーマ\n{topic}\n\n"
                    f"### 1. 提案役の最初の主張\n{proposer_text}\n\n"
                    f"### 2. 批判役の指摘\n{critic_text}\n\n"
                    f"### 3. 提案役の反論・補足\n{rebuttal_text}\n\n"
                    "以下の構成で最終出力を作成してください：\n"
                    "1. **【総合分析】議論の総括**\n"
                    "2. **【改善ポイント】回避すべきリスクと修正案**\n"
                    "3. **【最終完成版・アクションプラン】**"
                )
                # 審判フェーズはテキストのみで構成
                res_j1 = call_gemini_with_retry(
                    prompt_j1, sys_judge, temperature=0.3
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
            st.error(
                f"🚨 Gemini APIエラーが発生しました。アクセス集中による制限の可能性があります。数分待ってから再度お試しください: {e}"
            )
        except Exception as e:
            st.error(f"🚨 予期せぬエラーが発生しました: {e}")

# --------------------------------------------------
# 結果表示領域
# --------------------------------------------------
# (元のコードから変更なし)
if st.session_state.discussion_result:
    res = st.session_state.discussion_result

    st.divider()
    st.subheader("📊 議論結果および最終完成版")
    st.caption(f"実行日時: {res['timestamp']}")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏆 最終結論・完成版",
            "💡 1. 提案（肯定）",
            "⚡ 2. 批判（指摘）",
            "↩️ 3. 反論（対案）",
        ]
    )

    with tab1:
        st.success(res["judge"])

        full_markdown = (
            f"# AIマルチエージェント討論レポート\n"
            f"実行日時: {res['timestamp']}\n\n"
            f"## 検討テーマ\n{res['topic']}\n\n"
            f"---\n## 🏆 最終結論・完成版\n{res['judge']}\n\n"
            f"---\n## 💡 1. 提案役の主張\n{res['proposer']}\n\n"
            f"---\n## ⚡ 2. 批判役の指摘\n{res['critic']}\n\n"
            f"---\n## ↩️ 3. 提案役の反論・補足\n{res['rebuttal']}\n"
        )

        st.download_button(
            label="📝 討論レポートをダウンロード (.md)",
            data=full_markdown,
            file_name=f"discussion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tab2:
        st.info(res["proposer"])

    with tab3:
        st.warning(res["critic"])

    with tab4:
        st.markdown(res["rebuttal"])
