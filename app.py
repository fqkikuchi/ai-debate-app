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
    page_title="AIマルチエージェント討論 (真のマルチエージェント・自動待機版)",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 AIマルチエージェント討論システム (自動待機・安定版)")
st.caption(
    "Gemini 3.6 Flash による 💡提案役 vs ⚡批判役 vs ↩️反論 vs 🏆審判役 の4段階独立ディベート"
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

# すべての役割で上位モデル（Flash）を使用し、品質を最大化
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
    """API制限（429エラー）を検知し、自動で待機してリトライする関数"""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=800,
    )
    
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
        except APIError as e:
            error_msg = str(e)
            # 429エラー（Rate Limit）かどうかを判定
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt == max_retries - 1:
                    raise e # 最大リトライ回数を超えたらエラー終了
                
                # エラーメッセージから「Please retry in X.XXs.」の秒数を抽出
                wait_time = 20 # デフォルトの待機時間
                match = re.search(r"retry in ([\d\.]+)s", error_msg)
                if match:
                    wait_time = int(float(match.group(1))) + 2 # 念のため+2秒のバッファ

                status_container.warning(f"⚠️ 無料枠のAPI制限に到達しました。{wait_time}秒待機して自動再開します（{attempt+1}回目のリトライ）...")
                time.sleep(wait_time)
                status_container.info("🔄 処理を再開します...")
            else:
                # 429以外の致命的なエラー（認証エラーなど）は即座に停止
                raise e

def manual_wait(seconds: int, message: str, status_container):
    """通常のインターバル待機UI"""
    status_container.write(f"⏳ {message} (API制限回避のため {seconds}秒 待機中...)")
    time.sleep(seconds)

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
# 議論実行処理 (真のマルチエージェント・自動待機)
# --------------------------------------------------
if st.button("🚀 討論を開始する", type="primary", use_container_width=True):
    topic = sanitize_input(topic_raw)

    if not topic and processed_image is None:
        st.warning("テーマを入力するか、画像をアップロードしてください。")
    else:
        try:
            with st.status(
                "🤖 独立したAIエージェントたちが議論を開始します...",
                expanded=True,
            ) as status:
                
                # 基本のインターバル（連続アクセスを防ぐためのベース待機時間）
                BASE_WAIT = 15

                # --- PHASE 1: 提案役 ---
                status.write("1/4 💡 【提案役】強みと推進案を構築中...")
                sys_proposer = "あなたはこのアイデアや企画を成功させたい「熱心な提案役」です。画像がある場合は視覚的特徴も言語化し、強みや期待できる成果を論理的に主張してください。"
                
                contents_p1 = [processed_image, f"### 検討テーマ\n{topic}"] if processed_image else [f"### 検討テーマ\n{topic}"]
                res_p1 = call_gemini_with_smart_retry(
                    contents=contents_p1, system_instruction=sys_proposer, temperature=0.7, status_container=status
                )
                proposer_text = res_p1.text
                
                manual_wait(BASE_WAIT, "提案役の意見をまとめています", status)

                # --- PHASE 2: 批判役 ---
                status.write("2/4 ⚡ 【批判役】リスクと欠点を徹底検証中...")
                sys_critic = "あなたは徹底的な「批判役（悪魔の代弁者）」です。提案役の主張に対し、見落としている致命的なリスクや構造的欠点を冷静に3点以内で指摘してください。"
                
                prompt_c1 = f"### 検討テーマ\n{topic}\n\n### 提案役の主張\n{proposer_text}"
                res_c1 = call_gemini_with_smart_retry(
                    contents=[prompt_c1], system_instruction=sys_critic, temperature=0.2, status_container=status
                )
                critic_text = res_c1.text

                manual_wait(BASE_WAIT, "批判役の指摘を精査しています", status)

                # --- PHASE 3: 提案役の反論 ---
                status.write("3/4 ↩️ 【提案役】批判に対する誤解の解明・対案（反論）を作成中...")
                sys_rebuttal = "あなたは提案役です。批判役の指摘を真摯に受け止めつつ、批判を吸収した「現実的な補足・対案」を簡潔に提示してください。"
                
                prompt_r1 = f"### 批判役の指摘\n{critic_text}"
                res_r1 = call_gemini_with_smart_retry(
                    contents=[prompt_r1], system_instruction=sys_rebuttal, temperature=0.5, status_container=status
                )
                rebuttal_text = res_r1.text

                manual_wait(BASE_WAIT, "最終判断の準備をしています", status)

                # --- PHASE 4: 審判役 ---
                status.write("4/4 🏆 【審判役】すべての議論を統合し、最適解を策定中...")
                sys_judge = (
                    "あなたは公正な「審判（まとめ役）」です。"
                    "提案、批判、反論を精査し、リスクを最小化しつつ効果を最大化する【最終完成版】を作成してください。\n"
                    "出力構成：\n1. **【総合分析】**\n2. **【回避すべきリスク】**\n3. **【最終完成版・アクションプラン】**"
                )
                
                prompt_j1 = f"### 検討テーマ\n{topic}\n\n### 1. 提案\n{proposer_text}\n### 2. 批判\n{critic_text}\n### 3. 反論\n{rebuttal_text}"
                res_j1 = call_gemini_with_smart_retry(
                    contents=[prompt_j1], system_instruction=sys_judge, temperature=0.3, status_container=status
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
            st.error(f"🚨 APIエラーにより処理を中断しました: {e}")
        except Exception as e:
            st.error(f"🚨 予期せぬエラーが発生しました: {e}")

# --------------------------------------------------
# 結果表示領域
# --------------------------------------------------
if st.session_state.discussion_result:
    res = st.session_state.discussion_result
    st.divider()
    st.subheader("📊 議論結果および最終完成版")
    st.caption(f"実行日時: {res['timestamp']}")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 最終結論", "💡 提案", "⚡ 批判", "↩️ 反論"])
    with tab1:
        st.success(res["judge"])
    with tab2:
        st.info(res["proposer"])
    with tab3:
        st.warning(res["critic"])
    with tab4:
        st.markdown(res["rebuttal"])
