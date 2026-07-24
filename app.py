# --------------------------------------------------
# モデルの定義（2種類用意する）
# --------------------------------------------------
MODEL_LITE = "gemini-3.5-flash-lite"
MODEL_PRO  = "gemini-3.6-flash"

# --------------------------------------------------
# ユーティリティ関数（引数にモデルを追加）
# --------------------------------------------------
def call_gemini_with_retry(
    contents, system_instruction: str, temperature: float, model_name: str, max_retries: int = 3
):
    """APIコールとエラーハンドリング（モデルを動的に切り替え）"""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=600, 
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


# --- (中略: 入力フォーム領域などはそのまま) ---


                # --------------------------------------------------
                # 議論実行処理 (各フェーズでモデルを切り替える)
                # --------------------------------------------------
                
                # --- PHASE 1: 提案役（Liteでコスト削減） ---
                st.write("1/4 💡 【提案役】強みと推進案を構築中...")
                sys_proposer = "あなたはこのアイデアや企画を成功させたい「熱心な提案役」です。..."
                contents_p1 = [processed_image, f"### 検討テーマ\n{topic}"] if processed_image else [f"### 検討テーマ\n{topic}"]
                
                res_p1 = call_gemini_with_retry(
                    contents=contents_p1, system_instruction=sys_proposer, temperature=0.7, model_name=MODEL_LITE
                )
                proposer_text = res_p1.text
                wait_for_api_limit(WAIT_SECONDS, "提案役の意見をまとめています")


                # --- PHASE 2: 批判役（Proで鋭い指摘） ---
                st.write("2/4 ⚡ 【批判役】リスクと欠点を徹底検証中...")
                sys_critic = "あなたは徹底的な「批判役（悪魔の代弁者）」です。..."
                prompt_c1 = f"### 検討テーマ\n{topic}\n\n### 提案役の主張\n{proposer_text}"
                
                res_c1 = call_gemini_with_retry(
                    contents=[prompt_c1], system_instruction=sys_critic, temperature=0.2, model_name=MODEL_PRO
                )
                critic_text = res_c1.text
                wait_for_api_limit(WAIT_SECONDS, "批判役の指摘を精査しています")


                # --- PHASE 3: 提案役の反論（Liteで防御） ---
                st.write("3/4 ↩️ 【提案役】批判に対する誤解の解明・対案（反論）を作成中...")
                sys_rebuttal = "あなたは提案役です。批判役の指摘を受け止めつつ..."
                prompt_r1 = f"### 批判役の指摘\n{critic_text}"
                
                res_r1 = call_gemini_with_retry(
                    contents=[prompt_r1], system_instruction=sys_rebuttal, temperature=0.5, model_name=MODEL_LITE
                )
                rebuttal_text = res_r1.text
                wait_for_api_limit(WAIT_SECONDS, "最終判断の準備をしています")


                # --- PHASE 4: 審判役（Proで高品質な最終出力） ---
                st.write("4/4 🏆 【審判役】すべての議論を統合し、最適解を策定中...")
                sys_judge = "あなたは公正な「審判（まとめ役）」です。..."
                prompt_j1 = f"### 検討テーマ\n{topic}\n\n### 1. 提案\n{proposer_text}\n### 2. 批判\n{critic_text}\n### 3. 反論\n{rebuttal_text}\n\n"
                
                res_j1 = call_gemini_with_retry(
                    contents=[prompt_j1], system_instruction=sys_judge, temperature=0.3, model_name=MODEL_PRO
                )
                judge_text = res_j1.text

