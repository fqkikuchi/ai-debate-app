import streamlit as st
from google import genai
import os

# ページ設定
st.set_page_config(page_title="AIマルチエージェント討論", page_icon="🤖")
st.title("🤖 AIマルチエージェント討論システム")
st.caption("Geminiによる 提案役 vs 批判役 vs 審判役 の徹底議論")

# APIキーの取得（Streamlit Secretsから取得）
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("APIキーが設定されていません。StreamlitのSecretsにGEMINI_API_KEYを設定してください。")
    st.stop()

# 最新のGoogle GenAI SDK初期化
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

# 入力フォーム
topic = st.text_input("検討したいテーマやアイデアを入力してください:", placeholder="例：地方でAIを使ったSNS運用代行副業は稼げる？")

if st.button("🚀 議論を開始する", type="primary"):
    if not topic:
        st.warning("テーマを入力してください。")
    else:
        # --- 1. 提案役（肯定派） ---
        with st.spinner("1/3 提案役がポジティブなアイデアを作成中..."):
            prompt_proposer = f"""
            テーマ: 「{topic}」
            あなたはこのアイデアを成功させたい「提案役」です。
            このアイデアのメリット、具体的で現実的な実践方法、収益化の可能性を論理的に熱く主張してください。
            """
            res_proposer = client.models.generate_content(model=MODEL_NAME, contents=prompt_proposer)
            proposer_text = res_proposer.text

        st.subheader("💡 1. 提案役（肯定派）の主張")
        st.write(proposer_text)
        st.divider()

        # --- 2. 批判役（悪魔の代弁者） ---
        with st.spinner("2/3 批判役がリスクや盲点を徹底ツッコミ中..."):
            prompt_critic = f"""
            テーマ: 「{topic}」
            提案役の主張:
            {proposer_text}

            あなたは徹底的な「批判役（悪魔の代弁者）」です。
            提案役の甘い見通し、見落としているデメリットやリスク、実行する際の障壁や懸念点を、論理的かつ容赦なく指摘・批判してください。
            """
            res_critic = client.models.generate_content(model=MODEL_NAME, contents=prompt_critic)
            critic_text = res_critic.text

        st.subheader("⚡ 2. 批判役（悪魔の代弁者）のツッコミ")
        st.write(critic_text)
        st.divider()

        # --- 3. 審判役（最終結論） ---
        with st.spinner("3/3 審判役が両者の意見を統合し、現実解を導出中..."):
            prompt_judge = f"""
            テーマ: 「{topic}」
            【提案役の主張】: {proposer_text}
            【批判役の指摘】: {critic_text}

            あなたはずば抜けた洞察力を持つ「審判（まとめ役）」です。
            提案と批判の両方の言い分を冷徹に分析し、
            1. 提案役のどこが甘かったか
            2. 批判役のどのリスクを重く受け止めるべきか
            3. 最終的に「リスクを最小限にしつつ、現実的に成功させるための具体的なアクションプラン」
            を明快に結論づけてください。
            """
            res_judge = client.models.generate_content(model=MODEL_NAME, contents=prompt_judge)
            judge_text = res_judge.text

        st.subheader("🏆 3. 審判の最終結論・アクションプラン")
        st.success(judge_text)
