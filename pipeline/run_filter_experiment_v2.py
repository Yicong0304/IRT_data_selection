"""
IRT 筛选实验 v2：在 prompt 和 rater 两个维度上验证 IRT 参数的筛选能力。

实验设计:
  Part 1a: 仅筛 prompt（v1 评分: quality_score = α_p × λ_p，不含难度惩罚）
  Part 1b: 仅筛 prompt（v2 评分: quality_score = α_p × λ_p / (1 + |γ_p|)，含难度惩罚）
  Part 2:  筛 rater → 联合筛 prompt
           - rater ≤ 4 个: Leave-one-out 逐个去掉，选最优子集
           - rater > 4 个: 按 rater quality score = α_r / (1 + |β_r|) 排序，逐步淘汰

用法:
    python run_filter_experiment_v2.py             # 跑全部实验
    python run_filter_experiment_v2.py --ks 20 30 40 50 60 70 80
    python run_filter_experiment_v2.py --keep-raters 5   # 指定保留 rater 数量
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
JUDGE_RESULTS = "/root/zhaoyicong/IRT_data_selection/judge_results.json"
OUTPUT_DIR = "/root/zhaoyicong/IRT_data_selection/ChatbotIRT/Resultv2"

# rater 数量阈值：≤ 此值用 LOO，> 此值用 quality score 排序
RATER_LOO_THRESHOLD = 4


def load_data():
    """加载 judge 数据，返回适配 IRT 的 DataFrame"""
    with open(JUDGE_RESULTS, "r", encoding="utf-8") as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    df["rater_id"] = df["judge"]
    return df


def fit_irt_on_subset(df_full, prompt_subset, rater_subset=None, use_rater=True):
    """
    在指定的 prompt 子集（和可选的 rater 子集）上拟合 IRT，返回 model abilities DataFrame。
    """
    df_sub = df_full[df_full["prompt_id"].isin(prompt_subset)].copy()

    if rater_subset is not None:
        df_sub = df_sub[df_sub["rater_id"].isin(rater_subset)].copy()

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


def fit_full_irt(df, use_rater=True):
    """拟合完整 IRT 模型，返回 evaluator 对象（用于提取参数）"""
    if use_rater and df["rater_id"].nunique() >= 2:
        df_fit = df[["model_a", "model_b", "outcome", "rater_id", "prompt_id"]].copy()
        evaluator = PairwiseIREvaluator(
            use_model_ability=True,
            use_rater_disc=True, use_rater_bias=True,
            use_prompt_disc=True, use_prompt_diff=True,
            use_feasibility=True,
        )
    else:
        df_fit = df[["model_a", "model_b", "outcome", "prompt_id"]].copy()
        evaluator = PairwiseIREvaluator(
            use_model_ability=True,
            use_rater_disc=False, use_rater_bias=False,
            use_prompt_disc=True, use_prompt_diff=True,
            use_feasibility=True,
        )
    evaluator.fit(df_fit, num_epochs=3000, lr=0.03)
    return evaluator


def compute_metrics(baseline_abilities, subset_abilities):
    """
    计算筛选后 vs baseline 的指标:
    - spearman_rho: 排名相关性 (越接近1越好)
    - kendall_tau: 排名相关性
    - ability_std: 能力值标准差 (越大说明区分度越好)
    - max_gap: 最强与最弱模型的能力差
    """
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


def run_prompt_filter(df, all_prompts, baseline_abilities, baseline_std, baseline_gap,
                      prompt_params, ks, random_repeats, use_rater, rater_subset=None,
                      label_prefix=""):
    """
    对给定数据做 prompt 筛选实验，返回结果 DataFrame。
    rater_subset: 如果指定，只用这些 rater 的数据。
    """
    results_table = []

    for K in ks:
        if K > len(all_prompts):
            continue
        print(f"\n{'='*60}")
        print(f"📊 {label_prefix}K = {K}")
        print(f"{'='*60}")

        # --- IRT-filtered ---
        top_k_prompts = prompt_params.head(K)["prompt"].tolist()
        print(f"  [IRT-filtered] 拟合 top-{K} prompts...")
        irt_abilities = fit_irt_on_subset(df, top_k_prompts, rater_subset, use_rater)
        irt_metrics = compute_metrics(baseline_abilities, irt_abilities)
        irt_metrics["method"] = "IRT-filtered"
        irt_metrics["K"] = K
        results_table.append(irt_metrics)
        print(f"    Spearman ρ={irt_metrics['spearman_rho']:.3f}, "
              f"Kendall τ={irt_metrics['kendall_tau']:.3f}, "
              f"Std={irt_metrics['ability_std']:.4f}, "
              f"Gap={irt_metrics['max_gap']:.4f}")

        # --- Random baseline ---
        random_metrics_list = []
        for r in range(random_repeats):
            random_prompts = list(np.random.choice(all_prompts, K, replace=False))
            try:
                rand_abilities = fit_irt_on_subset(df, random_prompts, rater_subset, use_rater)
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

    # 组装结果
    results_df = pd.DataFrame(results_table)
    baseline_row = {
        "method": "Baseline (all)",
        "K": len(all_prompts),
        "spearman_rho": 1.0,
        "kendall_tau": 1.0,
        "ability_std": baseline_std,
        "max_gap": baseline_gap,
    }
    results_df = pd.concat([pd.DataFrame([baseline_row]), results_df], ignore_index=True)
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="*", type=int,
                        default=[20, 30, 40, 50, 60, 70, 80],
                        help="要测试的筛选数量 K")
    parser.add_argument("--random-repeats", type=int, default=10,
                        help="随机基线重复次数")
    parser.add_argument("--use-rater", action="store_true", default=True,
                        help="是否启用 rater 参数")
    parser.add_argument("--keep-raters", type=int, default=None,
                        help="保留的 rater 数量（默认: 去掉 1 个）")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    # ==========================================
    # 加载数据
    # ==========================================
    print("📥 加载数据...")
    df = load_data()
    all_prompts = sorted(df["prompt_id"].unique())
    all_raters = sorted(df["rater_id"].unique())
    print(f"   总 prompt 数: {len(all_prompts)}, 总比较数: {len(df)}")
    print(f"   Raters: {all_raters}")

    # ==========================================
    # Full baseline: 全部 prompt + 全部 rater
    # ==========================================
    print("\n🔧 [Baseline] 全部 prompt + 全部 rater 拟合 IRT...")
    baseline_abilities = fit_irt_on_subset(df, all_prompts, use_rater=args.use_rater)
    baseline_std = baseline_abilities["ability"].std()
    baseline_gap = baseline_abilities["ability"].max() - baseline_abilities["ability"].min()

    print("   Baseline Model Abilities:")
    print(baseline_abilities.to_string(index=False))
    print(f"   Ability Std: {baseline_std:.4f}, Max Gap: {baseline_gap:.4f}")

    # ==========================================
    # 拟合完整 IRT 获取参数
    # ==========================================
    print("\n🔧 拟合完整 IRT 获取 prompt 参数...")
    full_evaluator = fit_full_irt(df, use_rater=args.use_rater)
    prompt_params = full_evaluator.get_prompt_parameters()

    # 两种 quality_score:
    #   v1: α_p × λ_p（不含难度惩罚）
    #   v2: α_p × λ_p / (1 + |γ_p|)（含难度惩罚）
    prompt_params["quality_score_v1"] = (
        prompt_params["discriminability"] * prompt_params["feasibility"]
    )
    prompt_params["quality_score_v2"] = (
        prompt_params["discriminability"] * prompt_params["feasibility"]
        / (1 + prompt_params["difficulty_offset"].abs())
    )
    print(f"   Prompt quality score v1 (α×λ):         mean={prompt_params['quality_score_v1'].mean():.3f}, "
          f"std={prompt_params['quality_score_v1'].std():.3f}")
    print(f"   Prompt quality score v2 (α×λ/(1+|γ|)): mean={prompt_params['quality_score_v2'].mean():.3f}, "
          f"std={prompt_params['quality_score_v2'].std():.3f}")

    # 打印 rater 参数
    if full_evaluator.n_raters > 0:
        rater_params = full_evaluator.get_rater_parameters()
        print(f"\n📊 Rater Parameters:")
        print(rater_params.to_string(index=False))

    # ==========================================================
    # Part 1: 仅筛 prompt（全部 rater），两种评分方式
    # ==========================================================

    # --- Part 1a: v1 评分 (α_p × λ_p) ---
    print("\n" + "=" * 70)
    print("📋 Part 1a: 仅筛 Prompt（v1: α×λ，不含难度惩罚）")
    print("=" * 70)

    prompt_params_v1 = prompt_params.sort_values("quality_score_v1", ascending=False)
    part1a_results = run_prompt_filter(
        df, all_prompts, baseline_abilities, baseline_std, baseline_gap,
        prompt_params_v1, args.ks, args.random_repeats, args.use_rater,
        rater_subset=None, label_prefix="[Part1a] "
    )

    part1a_path = os.path.join(OUTPUT_DIR, "filter_v2_prompt_only_v1score.csv")
    part1a_results.to_csv(part1a_path, index=False)
    print(f"\n💾 Part 1a 结果 → {part1a_path}")
    print(part1a_results.to_string(index=False))

    # --- Part 1b: v2 评分 (α_p × λ_p / (1 + |γ_p|)) ---
    print("\n" + "=" * 70)
    print("📋 Part 1b: 仅筛 Prompt（v2: α×λ/(1+|γ|)，含难度惩罚）")
    print("=" * 70)

    prompt_params_v2 = prompt_params.sort_values("quality_score_v2", ascending=False)
    part1b_results = run_prompt_filter(
        df, all_prompts, baseline_abilities, baseline_std, baseline_gap,
        prompt_params_v2, args.ks, args.random_repeats, args.use_rater,
        rater_subset=None, label_prefix="[Part1b] "
    )

    part1b_path = os.path.join(OUTPUT_DIR, "filter_v2_prompt_only.csv")
    part1b_results.to_csv(part1b_path, index=False)
    print(f"\n💾 Part 1b 结果 → {part1b_path}")
    print(part1b_results.to_string(index=False))

    # ==========================================================
    # Part 2: 筛 rater
    #   rater ≤ 4: Leave-one-out
    #   rater > 4: 按 rater quality score 排序淘汰
    # ==========================================================
    n_raters = len(all_raters)
    keep_n = args.keep_raters if args.keep_raters else (n_raters - 1)
    keep_n = max(2, min(keep_n, n_raters - 1))  # 至少保留 2 个，至少去掉 1 个

    print("\n" + "=" * 70)
    print(f"📋 Part 2: 筛 Rater（共 {n_raters} 个，保留 {keep_n} 个）")
    print("=" * 70)

    if n_raters <= RATER_LOO_THRESHOLD:
        # ---------- 模式 A: Leave-One-Out ----------
        print(f"   策略: Leave-One-Out（rater 数 ≤ {RATER_LOO_THRESHOLD}）")

        loo_results = []
        for drop_rater in all_raters:
            remaining_raters = [r for r in all_raters if r != drop_rater]
            print(f"\n  🔧 去掉 [{drop_rater}]，保留 {remaining_raters}...")

            loo_abilities = fit_irt_on_subset(
                df, all_prompts, rater_subset=remaining_raters, use_rater=args.use_rater
            )
            loo_m = compute_metrics(baseline_abilities, loo_abilities)
            loo_m["dropped_rater"] = drop_rater
            loo_m["remaining_raters"] = str(remaining_raters)
            loo_results.append(loo_m)

            print(f"    Spearman ρ={loo_m['spearman_rho']:.3f}, "
                  f"Kendall τ={loo_m['kendall_tau']:.3f}, "
                  f"Std={loo_m['ability_std']:.4f}, "
                  f"Gap={loo_m['max_gap']:.4f}")

        loo_df = pd.DataFrame(loo_results)
        loo_path = os.path.join(OUTPUT_DIR, "filter_v2_rater_loo.csv")
        loo_df.to_csv(loo_path, index=False)
        print(f"\n💾 Rater LOO 结果 → {loo_path}")
        print(loo_df.to_string(index=False))

        # 选出去掉后 ability_std 最大的那个（即最差 judge）
        best_drop = loo_df.loc[loo_df["ability_std"].idxmax()]
        worst_rater = best_drop["dropped_rater"]
        best_raters = [r for r in all_raters if r != worst_rater]
        print(f"\n🏆 最差 rater: [{worst_rater}]，去掉后 ability_std 最大")
        print(f"   最优 rater 子集: {best_raters}")

    else:
        # ---------- 模式 B: Rater Quality Score 排序 ----------
        print(f"   策略: Rater Quality Score 排序（rater 数 > {RATER_LOO_THRESHOLD}）")

        rater_params = full_evaluator.get_rater_parameters()

        # rater quality score = α_r / (1 + |β_r|)
        if "discriminability" in rater_params.columns and "bias" in rater_params.columns:
            rater_params["quality_score"] = (
                rater_params["discriminability"]
                / (1 + rater_params["bias"].abs())
            )
        elif "discriminability" in rater_params.columns:
            rater_params["quality_score"] = rater_params["discriminability"]
        else:
            rater_params["quality_score"] = 1.0 / (1 + rater_params["bias"].abs())

        rater_params = rater_params.sort_values("quality_score", ascending=False)
        print(f"\n📊 Rater Quality Scores:")
        print(rater_params.to_string(index=False))

        best_raters = rater_params.head(keep_n)["rater"].tolist()
        dropped_raters = rater_params.tail(n_raters - keep_n)["rater"].tolist()
        print(f"\n🏆 保留 top-{keep_n} raters: {best_raters}")
        print(f"   淘汰: {dropped_raters}")

        # 保存 rater 排序结果
        rater_rank_path = os.path.join(OUTPUT_DIR, "filter_v2_rater_ranking.csv")
        rater_params.to_csv(rater_rank_path, index=False)
        print(f"💾 Rater ranking → {rater_rank_path}")

        # 也跑一下各子集的指标供参考
        loo_results = []
        for dropped in dropped_raters:
            remaining = [r for r in all_raters if r != dropped]
            loo_abilities = fit_irt_on_subset(
                df, all_prompts, rater_subset=remaining, use_rater=args.use_rater
            )
            loo_m = compute_metrics(baseline_abilities, loo_abilities)
            loo_m["dropped_rater"] = dropped
            loo_m["remaining_raters"] = str(remaining)
            loo_results.append(loo_m)
        if loo_results:
            loo_df = pd.DataFrame(loo_results)
            loo_path = os.path.join(OUTPUT_DIR, "filter_v2_rater_loo.csv")
            loo_df.to_csv(loo_path, index=False)

        worst_rater = dropped_raters[0] if len(dropped_raters) == 1 else str(dropped_raters)

    # ==========================================================
    # Part 2 续: 联合筛选（最优 rater 子集 + prompt 筛选）
    # ==========================================================
    print("\n" + "=" * 70)
    print(f"📋 Part 2 续: 联合筛选（去掉 [{worst_rater}] + 筛 prompt）")
    print("=" * 70)

    # 在最优 rater 子集上重新拟合 IRT 获取 prompt 参数
    df_best_raters = df[df["rater_id"].isin(best_raters)].copy()
    print(f"\n🔧 在 {best_raters} 数据上重新拟合 IRT 获取 prompt 参数...")
    joint_evaluator = fit_full_irt(df_best_raters, use_rater=args.use_rater)
    joint_prompt_params = joint_evaluator.get_prompt_parameters()
    joint_prompt_params["quality_score"] = (
        joint_prompt_params["discriminability"] * joint_prompt_params["feasibility"]
        / (1 + joint_prompt_params["difficulty_offset"].abs())
    )
    joint_prompt_params = joint_prompt_params.sort_values("quality_score", ascending=False)

    # 联合筛选的 baseline: 最优 rater 子集 + 全部 prompt
    joint_baseline_abilities = fit_irt_on_subset(
        df, all_prompts, rater_subset=best_raters, use_rater=args.use_rater
    )
    joint_baseline_std = joint_baseline_abilities["ability"].std()
    joint_baseline_gap = (joint_baseline_abilities["ability"].max()
                          - joint_baseline_abilities["ability"].min())

    joint_results = run_prompt_filter(
        df, all_prompts, baseline_abilities, baseline_std, baseline_gap,
        joint_prompt_params, args.ks, args.random_repeats, args.use_rater,
        rater_subset=best_raters, label_prefix="[Joint] "
    )

    joint_path = os.path.join(OUTPUT_DIR, "filter_v2_joint.csv")
    joint_results.to_csv(joint_path, index=False)
    print(f"\n💾 联合筛选结果 → {joint_path}")
    print(joint_results.to_string(index=False))

    # ==========================================================
    # 可视化（3 行: Part1a, Part1b, Part2 Joint）
    # ==========================================================
    print("\n📈 生成可视化...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    metric_names = ["spearman_rho", "ability_std", "max_gap"]
    metric_titles = ["Ranking Preservation (Spearman ρ)",
                     "Ability Std Dev", "Max Ability Gap"]
    metric_ylabels = ["Spearman ρ", "Std of θ", "θ_max - θ_min"]

    for row, (results_df, row_title) in enumerate([
        (part1a_results, "Part 1a: Prompt Filter (α×λ)"),
        (part1b_results, "Part 1b: Prompt Filter (α×λ/(1+|γ|))"),
        (joint_results, f"Part 2: Joint (drop [{worst_rater}] + Prompt Filter)")
    ]):
        for col, (metric, title, ylabel) in enumerate(
            zip(metric_names, metric_titles, metric_ylabels)
        ):
            ax = axes[row][col]

            for method, marker, color in [
                ("IRT-filtered", "o", "red"),
                ("Random", "x", "gray"),
            ]:
                sub = results_df[results_df["method"] == method]
                if sub.empty:
                    continue
                ax.plot(sub["K"], sub[metric], marker=marker, color=color,
                        label=method, linewidth=1.5)

            # baseline 参考线
            baseline_val = {"spearman_rho": 1.0,
                            "ability_std": baseline_std,
                            "max_gap": baseline_gap}[metric]
            ax.axhline(baseline_val, color="blue", linestyle="--",
                       alpha=0.5, label="Baseline (all)")

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Number of Prompts (K)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 行标题
        axes[row][0].annotate(
            row_title, xy=(0, 0.5),
            xytext=(-axes[row][0].yaxis.labelpad - 15, 0),
            xycoords=axes[row][0].yaxis.label, textcoords="offset points",
            fontsize=11, ha="right", va="center", fontweight="bold",
            rotation=90,
        )

    plt.suptitle("IRT-based Data Filtering Experiment v2", fontsize=14)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    fig_path = os.path.join(OUTPUT_DIR, "filter_experiment_v2.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 图表 → {fig_path}")

    # ==========================================================
    # 最终汇总
    # ==========================================================
    print("\n" + "=" * 70)
    print("📊 实验结果汇总")
    print("=" * 70)
    print("\n--- Part 1a: 仅筛 Prompt (α×λ) ---")
    print(part1a_results.to_string(index=False))
    print("\n--- Part 1b: 仅筛 Prompt (α×λ/(1+|γ|)) ---")
    print(part1b_results.to_string(index=False))
    print(f"\n--- Part 2: Rater 筛选 (保留 {best_raters}) ---")
    if 'loo_df' in dir():
        print(loo_df.to_string(index=False))
    print(f"\n--- Part 2 续: 联合筛选 (去掉 [{worst_rater}]) ---")
    print(joint_results.to_string(index=False))


if __name__ == "__main__":
    main()
