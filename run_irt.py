"""
IRT 实验主脚本：加载 judge 比较结果，拟合 Pairwise IRT 模型，筛选高质量 prompt

用法:
    python run_irt.py
    python run_irt.py --epochs 5000 --lr 0.02
    python run_irt.py --no-feasibility
"""

import argparse
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from irt_raw_code import PairwiseIREvaluator

# ==========================================
# 路径配置
# ==========================================
JUDGE_RESULTS = "/root/zhaoyicong/ChatbotIRT/judge_results.json"
OUTPUT_DIR = "/root/zhaoyicong/ChatbotIRT/Results"


def load_judge_data(path):
    """
    加载 LLM judge 比较结果，输出严格列顺序的 DataFrame。

    ⚠️ irt_raw_code.py 的 fit() 把 DataFrame 直接转 tensor，
    按列位置索引取值：col0=model_a, col1=model_b, col2=outcome,
    之后按 offset 依次取 rater_id（如有）、prompt_id（如有）。
    """
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    df["rater_id"] = df["judge"]

    # 多个小模型 judge → rater 参数可估
    # 严格列顺序: [model_a, model_b, outcome, rater_id, prompt_id]
    df = df[["model_a", "model_b", "outcome", "rater_id", "prompt_id"]]
    return df


def main():
    parser = argparse.ArgumentParser(description="Fit Pairwise IRT model on judge results")
    parser.add_argument("--input", default=JUDGE_RESULTS, help="judge_results.json 路径")
    parser.add_argument("--epochs", type=int, default=3000, help="SVI 训练轮数")
    parser.add_argument("--lr", type=float, default=0.03, help="学习率")
    parser.add_argument("--no-rater-disc", action="store_true", help="关闭 rater discriminability")
    parser.add_argument("--no-rater-bias", action="store_true", help="关闭 rater bias")
    parser.add_argument("--no-prompt-disc", action="store_true", help="关闭 prompt discriminability")
    parser.add_argument("--no-prompt-diff", action="store_true", help="关闭 prompt difficulty offset")
    parser.add_argument("--no-feasibility", action="store_true", help="关闭 prompt feasibility")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print("📥 加载 judge 比较结果...")
    df = load_judge_data(args.input)

    all_models = sorted(set(df["model_a"]) | set(df["model_b"]))
    raters = df["rater_id"].unique()
    print(f"   比较数: {len(df)}")
    print(f"   模型 ({len(all_models)}): {all_models}")
    print(f"   Prompt 数: {df['prompt_id'].nunique()}")
    print(f"   Judge/Rater ({len(raters)}): {raters}")
    print(f"   model_a 胜率: {df['outcome'].mean():.3f}")

    # 2. 配置 IRT 模型
    use_rater_disc = not args.no_rater_disc
    use_rater_bias = not args.no_rater_bias
    use_prompt_disc = not args.no_prompt_disc
    use_prompt_diff = not args.no_prompt_diff
    use_feasibility = not args.no_feasibility

    # 单 judge 时自动关闭 rater 参数
    if len(raters) < 2:
        use_rater_disc = False
        use_rater_bias = False
        # 单 rater 不需要 rater_id 列，调整列顺序
        df = df[["model_a", "model_b", "outcome", "prompt_id"]]
        print("   ⚠️  仅 1 个 judge，自动关闭 rater 参数")

    print(f"\n⚙️  IRT 配置:")
    print(f"   model_ability=True, rater_disc={use_rater_disc}, rater_bias={use_rater_bias}")
    print(f"   prompt_disc={use_prompt_disc}, prompt_diff={use_prompt_diff}, feasibility={use_feasibility}")
    print(f"   epochs={args.epochs}, lr={args.lr}")

    evaluator = PairwiseIREvaluator(
        use_model_ability=True,
        use_rater_disc=use_rater_disc,
        use_rater_bias=use_rater_bias,
        use_prompt_disc=use_prompt_disc,
        use_prompt_diff=use_prompt_diff,
        use_feasibility=use_feasibility,
    )

    # 3. 拟合
    print(f"\n🔧 开始训练...")
    evaluator.fit(df, num_epochs=args.epochs, lr=args.lr)

    # 4. 输出结果
    print("\n" + "=" * 50)
    print("📊 Model Ability Estimates (θ)")
    print("=" * 50)
    abilities = evaluator.get_abilities()
    print(abilities.to_string(index=False))

    if evaluator.n_prompts > 0:
        prompt_params = evaluator.get_prompt_parameters()
        print(f"\n📊 Prompt Parameters (共 {len(prompt_params)} 个)")
        print(prompt_params.describe().to_string())

    if evaluator.n_raters > 0:
        rater_params = evaluator.get_rater_parameters()
        print(f"\n📊 Rater Parameters")
        print(rater_params.to_string(index=False))

    # 5. 保存
    abilities.to_csv(os.path.join(OUTPUT_DIR, "model_abilities.csv"), index=False)
    print(f"\n💾 Model abilities → {OUTPUT_DIR}/model_abilities.csv")

    if evaluator.n_prompts > 0:
        prompt_params.to_csv(os.path.join(OUTPUT_DIR, "prompt_params.csv"), index=False)
        print(f"💾 Prompt params  → {OUTPUT_DIR}/prompt_params.csv")

    if evaluator.n_raters > 0:
        rater_params.to_csv(os.path.join(OUTPUT_DIR, "rater_params.csv"), index=False)
        print(f"💾 Rater params   → {OUTPUT_DIR}/rater_params.csv")

    # 6. 可视化
    try:
        fig = evaluator.visualize_parameters()
        fig_path = os.path.join(OUTPUT_DIR, "irt_parameters.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"📈 Visualization  → {fig_path}")
    except Exception as e:
        print(f"⚠️  可视化失败 (不影响结果): {e}")

    # 7. 模型间胜率预测矩阵
    print("\n📊 Pairwise Win Probability Matrix:")
    win_matrix = pd.DataFrame(index=all_models, columns=all_models, dtype=float)
    for m_a in all_models:
        for m_b in all_models:
            if m_a == m_b:
                win_matrix.loc[m_a, m_b] = 0.5
            else:
                win_matrix.loc[m_a, m_b] = evaluator.predict_win_probability(m_a, m_b)
    print(win_matrix.round(3).to_string())
    win_matrix.to_csv(os.path.join(OUTPUT_DIR, "win_probability_matrix.csv"))
    print(f"💾 Win matrix     → {OUTPUT_DIR}/win_probability_matrix.csv")


if __name__ == "__main__":
    main()
