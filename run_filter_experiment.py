"""
IRT 筛选实验：验证 IRT 参数能筛选出高质量 prompt，
使得用更少的题目就能保持模型排名且拉大能力差距。

实验设计:
  1. Full baseline: 全部 100 题拟合 IRT → 基准排名 & ability
  2. IRT-filtered: 按 prompt discriminability × feasibility 排序，取 top-K 题重新拟合
  3. Random baseline: 随机选 K 题重新拟合（重复多次取均值）
  4. 对比指标: 排名相关性 (Spearman)、ability 标准差、ability gap

用法:
    python run_irt.py                          # 先跑这个生成 IRT 参数
    python run_filter_experiment.py            # 再跑筛选实验
    python run_filter_experiment.py --ks 20 30 50 70  # 自定义筛选数量
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from irt_raw_code import PairwiseIREvaluator

# ==========================================
# 路径配置
# ==========================================
JUDGE_RESULTS = "/root/zhaoyicong/ChatbotIRT/judge_results.json"
OUTPUT_DIR = "/root/zhaoyicong/ChatbotIRT/Results"


def load_data():
    """加载 judge 数据，返回适配 IRT 的 DataFrame"""
    with open(JUDGE_RESULTS, "r", encoding="utf-8") as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df["rater_id"] = df["judge"]
    return df


def fit_irt_on_subset(df_full, prompt_subset, use_rater):
    """
    在指定的 prompt 子集上拟合 IRT，返回 model abilities DataFrame。
    """
    df_sub = df_full[df_full["prompt_id"].isin(prompt_subset)].copy()

    if use_rater and df_sub["rater_id"].nunique() >= 2:
        df_sub = df_sub[["model_a", "model_b", "outcome", "rater_id", "prompt_id"]]
        evaluator = PairwiseIREvaluator(
            use_model_ability=True,
            use_rater_disc=True, use_rater_bias=True,
            use_prompt_disc=True, use_prompt_diff=True,
            use_feasibility=True,
        )
    else:
        df_sub = df_sub[["model_a", "model_b", "outcome", "prompt_id"]]
        evaluator = PairwiseIREvaluator(
            use_model_ability=True,
            use_rater_disc=False, use_rater_bias=False,
            use_prompt_disc=True, use_prompt_diff=True,
            use_feasibility=True,
        )

    evaluator.fit(df_sub, num_epochs=2000, lr=0.03)
    return evaluator.get_abilities()


def compute_metrics(baseline_abilities, subset_abilities):
    """
    计算筛选后 vs baseline 的指标:
    - spearman_rho: 排名相关性 (越接近1越好)
    - kendall_tau: 排名相关性
    - ability_std: 能力值标准差 (越大说明区分度越好)
    - max_gap: 最强与最弱模型的能力差
    """
    # 对齐模型顺序
    merged = baseline_abilities.merge(
        subset_abilities, on="model", suffixes=("_base", "_sub")
    )

    rho, _ = spearmanr(merged["ability_base"], merged["ability_sub"])
    tau, _ = kendalltau(merged["ability_base"], merged["ability_sub"])
    ability_std = subset_abilities["ability"].std()
    max_gap = subset_abilities["ability"].max() - subset_abilities["ability"].min()

    return {
        "spearman_rho": rho,
        "kendall_tau": tau,
        "ability_std": ability_std,
        "max_gap": max_gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="*", type=int, default=[20, 30, 50, 70],
                        help="要测试的筛选数量 K")
    parser.add_argument("--random-repeats", type=int, default=10,
                        help="随机基线重复次数")
    parser.add_argument("--use-rater", action="store_true", default=True,
                        help="是否启用 rater 参数")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    # 1. 加载数据
    print("📥 加载数据...")
    df = load_data()
    all_prompts = sorted(df["prompt_id"].unique())
    print(f"   总 prompt 数: {len(all_prompts)}, 总比较数: {len(df)}")

    # 2. Full baseline: 全部 100 题拟合
    print("\n🔧 [Baseline] 全部 prompt 拟合 IRT...")
    baseline_abilities = fit_irt_on_subset(df, all_prompts, args.use_rater)
    baseline_std = baseline_abilities["ability"].std()
    baseline_gap = baseline_abilities["ability"].max() - baseline_abilities["ability"].min()

    print("   Baseline Model Abilities:")
    print(baseline_abilities.to_string(index=False))
    print(f"   Ability Std: {baseline_std:.4f}, Max Gap: {baseline_gap:.4f}")

    # 3. 拟合一次完整 IRT 获取 prompt 参数用于筛选
    print("\n🔧 获取 prompt 参数用于筛选...")
    if args.use_rater and df["rater_id"].nunique() >= 2:
        df_full = df[["model_a", "model_b", "outcome", "rater_id", "prompt_id"]].copy()
        full_evaluator = PairwiseIREvaluator(
            use_model_ability=True,
            use_rater_disc=True, use_rater_bias=True,
            use_prompt_disc=True, use_prompt_diff=True,
            use_feasibility=True,
        )
    else:
        df_full = df[["model_a", "model_b", "outcome", "prompt_id"]].copy()
        full_evaluator = PairwiseIREvaluator(
            use_model_ability=True,
            use_rater_disc=False, use_rater_bias=False,
            use_prompt_disc=True, use_prompt_diff=True,
            use_feasibility=True,
        )
    full_evaluator.fit(df_full, num_epochs=3000, lr=0.03)
    prompt_params = full_evaluator.get_prompt_parameters()

    # 计算 prompt 质量分 = discriminability × feasibility
    prompt_params["quality_score"] = prompt_params["discriminability"] * prompt_params["feasibility"]
    prompt_params = prompt_params.sort_values("quality_score", ascending=False)
    print(f"   Prompt quality score 分布: mean={prompt_params['quality_score'].mean():.3f}, "
          f"std={prompt_params['quality_score'].std():.3f}")

    # 4. 对每个 K 值做实验
    results_table = []

    for K in args.ks:
        if K > len(all_prompts):
            continue
        print(f"\n{'='*60}")
        print(f"📊 K = {K}")
        print(f"{'='*60}")

        # --- IRT-filtered ---
        top_k_prompts = prompt_params.head(K)["prompt"].tolist()
        print(f"  [IRT-filtered] 拟合 top-{K} prompts...")
        irt_abilities = fit_irt_on_subset(df, top_k_prompts, args.use_rater)
        irt_metrics = compute_metrics(baseline_abilities, irt_abilities)
        irt_metrics["method"] = "IRT-filtered"
        irt_metrics["K"] = K
        results_table.append(irt_metrics)
        print(f"    Spearman ρ={irt_metrics['spearman_rho']:.3f}, "
              f"Kendall τ={irt_metrics['kendall_tau']:.3f}, "
              f"Std={irt_metrics['ability_std']:.4f}, "
              f"Gap={irt_metrics['max_gap']:.4f}")

        # --- Random baseline (多次取均值) ---
        random_metrics_list = []
        for r in range(args.random_repeats):
            random_prompts = list(np.random.choice(all_prompts, K, replace=False))
            try:
                rand_abilities = fit_irt_on_subset(df, random_prompts, args.use_rater)
                rand_m = compute_metrics(baseline_abilities, rand_abilities)
                random_metrics_list.append(rand_m)
            except Exception:
                continue

        if random_metrics_list:
            avg_random = {
                "method": "Random",
                "K": K,
                "spearman_rho": np.mean([m["spearman_rho"] for m in random_metrics_list]),
                "kendall_tau": np.mean([m["kendall_tau"] for m in random_metrics_list]),
                "ability_std": np.mean([m["ability_std"] for m in random_metrics_list]),
                "max_gap": np.mean([m["max_gap"] for m in random_metrics_list]),
            }
            results_table.append(avg_random)
            print(f"  [Random avg x{len(random_metrics_list)}] "
                  f"Spearman ρ={avg_random['spearman_rho']:.3f}, "
                  f"Kendall τ={avg_random['kendall_tau']:.3f}, "
                  f"Std={avg_random['ability_std']:.4f}, "
                  f"Gap={avg_random['max_gap']:.4f}")

    # 5. 汇总结果
    results_df = pd.DataFrame(results_table)

    # 加入 baseline 行
    baseline_row = {
        "method": "Baseline (all)",
        "K": len(all_prompts),
        "spearman_rho": 1.0,
        "kendall_tau": 1.0,
        "ability_std": baseline_std,
        "max_gap": baseline_gap,
    }
    results_df = pd.concat([pd.DataFrame([baseline_row]), results_df], ignore_index=True)

    print("\n" + "=" * 60)
    print("📊 实验结果汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(OUTPUT_DIR, "filter_experiment_results.csv"), index=False)
    print(f"\n💾 结果保存至 {OUTPUT_DIR}/filter_experiment_results.csv")

    # 6. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for method, marker, color in [("IRT-filtered", "o", "red"), ("Random", "x", "gray")]:
        sub = results_df[results_df["method"] == method]
        if sub.empty:
            continue
        axes[0].plot(sub["K"], sub["spearman_rho"], marker=marker, color=color, label=method)
        axes[1].plot(sub["K"], sub["ability_std"], marker=marker, color=color, label=method)
        axes[2].plot(sub["K"], sub["max_gap"], marker=marker, color=color, label=method)

    # baseline 参考线
    axes[0].axhline(1.0, color="blue", linestyle="--", alpha=0.5, label="Baseline")
    axes[1].axhline(baseline_std, color="blue", linestyle="--", alpha=0.5, label="Baseline")
    axes[2].axhline(baseline_gap, color="blue", linestyle="--", alpha=0.5, label="Baseline")

    axes[0].set_title("Ranking Preservation (Spearman ρ)")
    axes[0].set_xlabel("Number of Prompts (K)")
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend()

    axes[1].set_title("Ability Std Dev")
    axes[1].set_xlabel("Number of Prompts (K)")
    axes[1].set_ylabel("Std of θ")
    axes[1].legend()

    axes[2].set_title("Max Ability Gap")
    axes[2].set_xlabel("Number of Prompts (K)")
    axes[2].set_ylabel("θ_max - θ_min")
    axes[2].legend()

    plt.suptitle("IRT-based Prompt Filtering vs Random Selection", fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, "filter_experiment.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 图表保存至 {fig_path}")


if __name__ == "__main__":
    main()
