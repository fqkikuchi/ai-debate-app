import os
from google import genai
import streamlit as st

# --------------------------------------------------
# ページ初期設定
# --------------------------------------------------
st.set_page_config(
    page_title="AIマルチエージェント討論",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AIマルチエージェント討論システム")
st.caption("Geminiによる 提案役(肯定) vs 批判役(悪魔の代弁者) vs 審判役(統合) の徹底議論")

# APIキーの取得（Streamlit Secretsまたは環境変数から取得）
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("APIキーが設定されていません。StreamlitのSecretsまたは環境変数に GEMINI_API_KEY を設定してください。")
    st.stop()

# 最新のGoogle GenAI SDK初期化
client = genai.Client(api_key=api_key)

# モデルの指定（安定して動作する gemini-2.0-flash または gemini-1.5-flash を推奨）
MODEL_NAME = "gemini-3.6-flash"

# --------------------------------------------------
# 入力フォーム領域（ご要望の複数行入力対応）
# --------------------------------------------------
topic = st.text_area(
    "検討したいテーマ、アイデア、文章の下書きなどを入力してください:",
    placeholder=(
        "【例1：ビジネスアイデア】\n"
        "地方でAIを使ったSNS運用代行副業は稼げる？\n\n"
        "【例2：ラブレター・文章チェック】\n"
        "気になっている同僚に送りたいLINEメッセージの下書き：\n"
        "「〇〇さん、今週もお疲れ様です！もしよかったら金曜の夜に新しくできたイタリアンに行きませんか？」"
    ),
    height=180,  # 一目で確認できるよう高さを確保
)

# セッション状態（結果保持用）の初期化
if "discussion_result" not in st.session_state:
    st.session_state.discussion_result = None

# --------------------------------------------------
# 議論実行処理
# --------------------------------------------------
if st.button("🚀 議論を開始する", type="primary"):
    if not topic.strip():
        st.warning("テーマや文章を入力してください。")
    else:
        with st.status("🤖 AIエージェントたちが議論を進行中...", expanded=True) as status:

            # --- 1. 提案役（肯定派） ---
            st.write("1/3 💡 提案役がポジティブな視点で分析・作成中...")
            prompt_proposer = f"""
            テーマ/入力文:
            「{topic}」

            あなたはこのアイデアや文章を成功させたい「熱心な提案役（肯定派）」です。
            この内容の強み・メリット、具体的で現実的な実践方法、期待できる成果（ビジネスなら収益性や集客効果、個人的な文章・ラブレターなら相手への感情的アピールや成功率）を論理的かつ熱意を持って主張してください。
            """
            res_proposer = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_proposer
            )
            proposer_text = res_proposer.text

            # --- 2. 批判役（悪魔の代弁者） ---
            st.write("2/3 ⚡ 批判役がリスクや盲点を徹底ツッコミ中...")
            prompt_critic = f"""
            テーマ/入力文:
            「{topic}」

            【提案役の主張】:
            {proposer_text}

            あなたは徹底的な「批判役（悪魔の代弁者）」です。
            提案役の甘い見通し、見落としているデメリットやリスク、実行・送信する際の障壁や懸念点を、論理的かつ容赦なく指摘・批判してください。
            ※ラブレターや個人的メッセージの場合は「相手に引かれるリスク」「重すぎる/不自然な表現」「逆効果になるポイント」を中心に批判してください。
            """
            res_critic = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_critic
            )
            critic_text = res_critic.text

            # --- 3. 審判役（最終結論） ---
            st.write("3/3 🏆 審判役が両者の意見を統合し、最終解を導出中...")
            prompt_judge = f"""
            テーマ/入力文:
            「{topic}」

            【提案役の主張】:
            {proposer_text}

            【批判役の指摘】:
            {critic_text}

            あなたはずば抜けた洞察力を持つ「審判（まとめ役）」です。
            提案役と批判役のやり取りを踏まえ、以下の項目を明快に出力してください。

            1. **【分析】提案役の評価できる点 と 批判役の重く受け止めるべき指摘**
            2. **【改善の方向性】リスクを回避し、目的を達成するためのポイント**
            3. **【最終完成版・アクションプラン】**
               - ビジネス/企画の場合：リスクを抑えた現実的な「実行アクションプラン」
               - ラブレター/文章の場合：批判を踏まえてブラッシュアップした「そのまま使える最終完成文」
            """
            res_judge = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_judge
            )
            judge_text = res_judge.text

            status.update(label="✅ 議論が完了しました！", state="complete", expanded=False)

        # 結果をセッションに保存（再描画時のデータ消失防止）
        st.session_state.discussion_result = {
            "topic": topic,
            "proposer": proposer_text,
            "critic": critic_text,
            "judge": judge_text,
        }

# --------------------------------------------------
# 結果表示領域（タブ表示）
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

        # ダウンロード機能
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
