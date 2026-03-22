"""
从 Chatbot Arena 原始数据中筛选 200 条高质量 prompt，用于 IRT 实验。

筛选条件：
  1. 英文、单轮对话
  2. prompt 长度 20-500 字符
  3. 有明确胜负（排除 tie）
  4. 涉及目标模型的对比
  5. 去重、排除有害内容
  6. 按长度分桶采样保证多样性

用法:
    python select_prompts.py                  # 默认筛 200 条
    python select_prompts.py --n 300          # 筛 300 条
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

# ==========================================
# 配置
# ==========================================
PARQUET_PATH = "/root/zhaoyicong/Dataset/chatbot_arena_conversations/data/train-00000-of-00001-cced8514c7ed782a.parquet"
OUTPUT_PATH = "/root/zhaoyicong/ChatbotIRT/irt_experiment_prompts_200.json"

TARGET_MODELS = {"vicuna-13b", "wizardlm-13b", "koala-13b", "alpaca-13b"}

# prompt 长度范围
MIN_LEN = 20
MAX_LEN = 500


def extract_prompt(conversation):
    """从 conversation 数组中提取第一轮用户输入"""
    if conversation is None:
        return None
    for turn in conversation:
        if isinstance(turn, dict) and turn.get("role") == "user":
            return turn.get("content", "").strip()
    return None


def is_toxic(row):
    """检查是否有有害内容标记"""
    # toxic_chat_tag
    tag = row.get("toxic_chat_tag")
    if tag and isinstance(tag, (dict, str)):
        if isinstance(tag, dict):
            for v in tag.values():
                if v:
                    return True
        elif tag not in ("", "none", "None"):
            return True

    # openai_moderation
    mod = row.get("openai_moderation")
    if mod and isinstance(mod, dict):
        flagged = mod.get("flagged", False)
        if flagged:
            return True
        categories = mod.get("categories", {})
        if isinstance(categories, dict):
            for v in categories.values():
                if v:
                    return True
    elif mod and isinstance(mod, list):
        for item in mod:
            if isinstance(item, dict):
                if item.get("flagged", False):
                    return True
                categories = item.get("categories", {})
                if isinstance(categories, dict):
                    for v in categories.values():
                        if v:
                            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="筛选的 prompt 数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", default=OUTPUT_PATH, help="输出路径")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. 加载数据
    print("📥 加载 Chatbot Arena 数据...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"   总数据量: {len(df)}")

    # 2. 基础过滤
    print("\n🔧 开始筛选...")

    # 英文
    df = df[df["language"] == "English"].copy()
    print(f"   英文: {len(df)}")

    # 单轮
    df = df[df["turn"] == 1].copy()
    print(f"   单轮: {len(df)}")

    # 涉及目标模型
    df = df[
        df["model_a"].isin(TARGET_MODELS) & df["model_b"].isin(TARGET_MODELS)
    ].copy()
    print(f"   目标模型间对比: {len(df)}")

    # 有明确胜负
    df = df[df["winner"].isin(["model_a", "model_b"])].copy()
    print(f"   有明确胜负: {len(df)}")

    # 3. 提取 prompt 文本
    df["prompt_text"] = df["conversation_a"].apply(extract_prompt)
    df = df[df["prompt_text"].notna()].copy()
    df["prompt_len"] = df["prompt_text"].str.len()

    # 长度过滤
    df = df[(df["prompt_len"] >= MIN_LEN) & (df["prompt_len"] <= MAX_LEN)].copy()
    print(f"   长度 {MIN_LEN}-{MAX_LEN} 字符: {len(df)}")

    # 4. 排除有害内容
    toxic_mask = df.apply(is_toxic, axis=1)
    n_toxic = toxic_mask.sum()
    df = df[~toxic_mask].copy()
    print(f"   排除有害内容 ({n_toxic} 条): {len(df)}")

    # 5. 按 prompt 文本去重（保留第一条）
    df = df.drop_duplicates(subset="prompt_text", keep="first").copy()
    print(f"   去重后: {len(df)}")

    if len(df) < args.n:
        print(f"\n⚠️  筛选后只有 {len(df)} 条，不足 {args.n} 条。将使用全部 {len(df)} 条。")
        selected = df
    else:
        # 6. 按长度分桶采样，保证多样性
        bins = [MIN_LEN, 50, 100, 200, 300, MAX_LEN + 1]
        labels = ["20-50", "50-100", "100-200", "200-300", "300-500"]
        df["len_bin"] = pd.cut(df["prompt_len"], bins=bins, labels=labels, right=False)

        bin_counts = df["len_bin"].value_counts()
        print(f"\n📊 长度分桶分布:")
        for label in labels:
            cnt = bin_counts.get(label, 0)
            print(f"   {label}: {cnt} 条")

        # 按比例从各桶采样
        selected_parts = []
        total_available = len(df)
        for label in labels:
            bin_df = df[df["len_bin"] == label]
            if len(bin_df) == 0:
                continue
            # 按该桶占总量的比例分配名额，至少 1 条
            quota = max(1, int(args.n * len(bin_df) / total_available))
            quota = min(quota, len(bin_df))
            sampled = bin_df.sample(n=quota, random_state=args.seed)
            selected_parts.append(sampled)

        selected = pd.concat(selected_parts)

        # 如果不够，从剩余中补
        if len(selected) < args.n:
            remaining = df[~df.index.isin(selected.index)]
            extra = remaining.sample(
                n=min(args.n - len(selected), len(remaining)),
                random_state=args.seed
            )
            selected = pd.concat([selected, extra])

        # 如果超了，截断
        if len(selected) > args.n:
            selected = selected.sample(n=args.n, random_state=args.seed)

    print(f"\n✅ 最终筛选: {len(selected)} 条")

    # 7. 构造输出格式
    output = {}
    for _, row in selected.iterrows():
        qid = row["question_id"]
        if isinstance(qid, (np.integer, int)):
            qid = str(qid)

        winner_model = row["model_a"] if row["winner"] == "model_a" else row["model_b"]
        loser_model = row["model_b"] if row["winner"] == "model_a" else row["model_a"]

        output[qid] = {
            "prompt": row["prompt_text"],
            "human_battles": [
                {"winner": winner_model, "loser": loser_model}
            ]
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"💾 保存至 {args.output}")
    print(f"   共 {len(output)} 条 prompt")

    # 8. 统计摘要
    lens = selected["prompt_len"]
    print(f"\n📊 筛选结果统计:")
    print(f"   长度: mean={lens.mean():.0f}, median={lens.median():.0f}, "
          f"min={lens.min()}, max={lens.max()}")

    model_pairs = selected.apply(
        lambda r: tuple(sorted([r["model_a"], r["model_b"]])), axis=1
    )
    print(f"   模型对分布:")
    for pair, cnt in model_pairs.value_counts().items():
        print(f"     {pair[0]} vs {pair[1]}: {cnt}")


if __name__ == "__main__":
    main()
