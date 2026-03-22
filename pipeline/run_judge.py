"""
LLM-as-a-Judge: 用本地多个小模型对所有答题模型的回答做成对比较
生成的结果格式直接适配 irt_raw_code.py 中 PairwiseIREvaluator.fit() 的输入要求

用法:
    python run_judge.py                          # 用全部 3 个 judge 跑全部比较
    python run_judge.py --judges qwen mistral    # 只用指定的 judge
    python run_judge.py --dry-run                # 只看统计，不实际推理
"""

import json
import os
import gc
import random
import argparse
from itertools import combinations

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# ==========================================
# 配置区
# ==========================================
JUDGES_ROOT = "/root/zhaoyicong/judges"
JUDGE_CONFIGS = {
    "qwen":     os.path.join(JUDGES_ROOT, "Qwen2.5-7B-Instruct"),
    "mistral":  os.path.join(JUDGES_ROOT, "Mistral-7B-Instruct-v0.3"),
    "internlm": os.path.join(JUDGES_ROOT, "internlm2_5-7b-chat"),
}

ROLLOUT_DIR = "/root/zhaoyicong/IRT_data_selection/Rollout"
OUTPUT_FILE = "/root/zhaoyicong/IRT_data_selection/judge_results.json"

TARGET_MODELS = ["vicuna-13b", "wizardlm-13b", "koala-13b", "alpaca-13b"]

JUDGE_PROMPT_TEMPLATE = """Please compare the following two AI responses to the user's question.
Evaluate based on: helpfulness, accuracy, relevance, depth, and clarity.

[User Question]
{question}

[Response A]
{response_a}

[Response B]
{response_b}

Which response is better overall? Answer with ONLY a single letter: "A" or "B"."""


def load_all_responses():
    """加载所有答题模型的推理结果"""
    all_responses = {}
    for model_name in TARGET_MODELS:
        path = os.path.join(ROLLOUT_DIR, f"{model_name}_responses.json")
        if not os.path.exists(path):
            print(f"  ⚠️  缺少 {path}，跳过 {model_name}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            all_responses[model_name] = json.load(f)
        print(f"  ✅ {model_name}: {len(all_responses[model_name])} 条")
    return all_responses


def build_judge_inputs(all_responses, common_qids, pairs, existing_done, judge_name):
    """
    构建当前 judge 需要评判的所有输入。
    返回 (batch_prompts, batch_meta) 用于 vLLM 批量推理。
    """
    random.seed(42)  # 保证 swap 可复现

    batch_prompts = []
    batch_meta = []

    for qid in common_qids:
        question = all_responses[list(all_responses.keys())[0]][qid]["prompt"]
        for m_a, m_b in pairs:
            # 跳过已完成的
            if (qid, m_a, m_b, judge_name) in existing_done:
                # 需要消耗同一个 random 状态以保持 swap 一致性
                random.random()
                continue

            resp_a = all_responses[m_a][qid]["response"]
            resp_b = all_responses[m_b][qid]["response"]

            swapped = random.random() < 0.5

            if swapped:
                prompt_text = JUDGE_PROMPT_TEMPLATE.format(
                    question=question, response_a=resp_b, response_b=resp_a
                )
            else:
                prompt_text = JUDGE_PROMPT_TEMPLATE.format(
                    question=question, response_a=resp_a, response_b=resp_b
                )

            batch_prompts.append(prompt_text)
            batch_meta.append({
                "qid": qid,
                "m_a": m_a,
                "m_b": m_b,
                "swapped": swapped,
            })

    return batch_prompts, batch_meta


def format_chat_prompt(judge_name, user_text):
    """为不同 judge 模型构造 chat 格式的 prompt"""
    system = "You are a fair, impartial judge of AI assistant responses."

    if "qwen" in judge_name:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif "mistral" in judge_name:
        return f"[INST] {system}\n\n{user_text} [/INST]"
    elif "internlm" in judge_name:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return f"{system}\n\nUser: {user_text}\nAssistant:"


def parse_verdict(text):
    """从 judge 输出中提取 A 或 B"""
    text = text.strip().upper()
    # 直接就是 A 或 B
    if text in ("A", "B"):
        return text
    # 开头是 A 或 B
    if text and text[0] in ("A", "B"):
        return text[0]
    return None


def run_judge(judge_name, judge_path, all_responses, common_qids, pairs, results, done):
    """用单个 judge 模型跑所有比较"""
    batch_prompts, batch_meta = build_judge_inputs(
        all_responses, common_qids, pairs, done, judge_name
    )

    if not batch_prompts:
        print(f"  ⏭️  {judge_name}: 所有比较已完成，跳过")
        return results

    print(f"\n{'='*50}")
    print(f"🧑‍⚖️ Judge: {judge_name} ({judge_path})")
    print(f"   待评判: {len(batch_prompts)} 条")
    print(f"{'='*50}")

    # 格式化为 chat prompt
    formatted = [format_chat_prompt(judge_name, p) for p in batch_prompts]

    # 加载模型
    sampling_params = SamplingParams(temperature=0, max_tokens=3)
    llm = LLM(
        model=judge_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
    )

    # 批量推理
    print(f"  ⚡ 批量推理中...")
    outputs = llm.generate(formatted, sampling_params)

    # 解析结果
    new_count = 0
    skip_count = 0
    for meta, output in zip(batch_meta, outputs):
        verdict = parse_verdict(output.outputs[0].text)
        if verdict is None:
            skip_count += 1
            continue

        # 还原 swap
        if meta["swapped"]:
            outcome = 0 if verdict == "A" else 1
        else:
            outcome = 1 if verdict == "A" else 0

        results.append({
            "prompt_id": meta["qid"],
            "model_a": meta["m_a"],
            "model_b": meta["m_b"],
            "outcome": outcome,
            "judge": judge_name,
        })
        new_count += 1

    print(f"  ✅ {judge_name}: {new_count} 条有效结果, {skip_count} 条无法解析")

    # 中间保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  💾 已保存 (累计 {len(results)} 条)")

    # 释放显存
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judges", nargs="*", default=None,
        help=f"指定 judge，可选: {list(JUDGE_CONFIGS.keys())}，默认全部"
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计不推理")
    args = parser.parse_args()

    judges = args.judges or list(JUDGE_CONFIGS.keys())
    for j in judges:
        if j not in JUDGE_CONFIGS:
            print(f"❌ 未知 judge: {j}，可选: {list(JUDGE_CONFIGS.keys())}")
            return

    print(f"📂 加载答题模型的回答...")
    all_responses = load_all_responses()
    available_models = list(all_responses.keys())

    if len(available_models) < 2:
        print("❌ 至少需要 2 个模型的结果")
        return

    common_qids = sorted(
        set.intersection(*[set(all_responses[m].keys()) for m in available_models])
    )
    pairs = list(combinations(available_models, 2))
    per_judge = len(common_qids) * len(pairs)
    total = per_judge * len(judges)

    print(f"📊 {len(common_qids)} prompts × {len(pairs)} pairs × {len(judges)} judges = {total} comparisons")

    if args.dry_run:
        for j in judges:
            print(f"  - {j}: {JUDGE_CONFIGS[j]}")
        print("(dry-run, exiting)")
        return

    # 加载已有进度
    results = []
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        done = {(r["prompt_id"], r["model_a"], r["model_b"], r["judge"]) for r in results}
        print(f"⏭️  已有 {len(results)} 条结果")

    # 逐个 judge 跑
    for judge_name in judges:
        judge_path = JUDGE_CONFIGS[judge_name]
        if not os.path.exists(judge_path):
            print(f"  ⚠️  {judge_name} 模型路径不存在: {judge_path}，跳过")
            continue
        results = run_judge(
            judge_name, judge_path, all_responses, common_qids, pairs, results, done
        )

    print(f"\n🎉 全部完成！共 {len(results)} 条比较结果，保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
