import json
import os
import sys
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# ==========================================
# 模型名 -> 本地路径的映射（与 Models/ 目录实际名称对齐）
# ==========================================
MODELS_ROOT = "/root/zhaoyicong/Models"
MODEL_CONFIGS = {
    "vicuna-13b":   os.path.join(MODELS_ROOT, "vicuna-13b-v1.3"),
    "wizardlm-13b": os.path.join(MODELS_ROOT, "WizardLM-13B-V1.2"),
    "koala-13b":    os.path.join(MODELS_ROOT, "koala-13B-HF"),
    "alpaca-13b":   os.path.join(MODELS_ROOT, "alpaca-13b"),
    # chatglm-6b: 第一代 GLM tokenizer 与当前 transformers/vLLM 不兼容，跳过
    # "chatglm-6b":   os.path.join(MODELS_ROOT, "chatglm-6b"),
}

INPUT_JSON = "/root/zhaoyicong/ChatbotIRT/irt_experiment_prompts.json"
OUTPUT_DIR = "/root/zhaoyicong/ChatbotIRT/Rollout"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_prompt(model_name, query):
    """为不同时代的模型套用它们当年最熟悉的对话模板"""
    if "vicuna" in model_name or "wizardlm" in model_name:
        return f"USER: {query}\nASSISTANT:"
    elif "alpaca" in model_name:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{query}\n\n### Response:\n"
        )
    elif "koala" in model_name:
        return f"BEGINNING OF CONVERSATION: USER: {query} GPT:"
    elif "chatglm" in model_name:
        return f"[Round 1]\n\n问：{query}\n\n答："
    else:
        return query


def run_single_model(model_name, model_path, qids, raw_prompts):
    """对单个模型执行推理并保存结果"""
    output_file = os.path.join(OUTPUT_DIR, f"{model_name}_responses.json")

    # 跳过已完成的模型
    if os.path.exists(output_file):
        print(f"⏭️  {model_name} 已有结果，跳过。删除 {output_file} 可重跑。")
        return

    print(f"\n{'='*50}")
    print(f"🚀 加载模型: {model_name} ({model_path})")
    print(f"{'='*50}")

    formatted_prompts = [format_prompt(model_name, p) for p in raw_prompts]

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
    )

    print("⚡ 开始批量生成回答...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 整理结果
    model_results = {}
    for i, (qid, output) in enumerate(zip(qids, outputs)):
        model_results[qid] = {
            "prompt": raw_prompts[i],
            "response": output.outputs[0].text.strip(),
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=4, ensure_ascii=False)

    print(f"✅ {model_name} 完成，结果保存至 {output_file}")

    # 释放显存，为下一个模型腾空间
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # 支持命令行指定单个模型: python run_rollout.py vicuna-13b
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
        for t in targets:
            if t not in MODEL_CONFIGS:
                print(f"❌ 未知模型: {t}，可选: {list(MODEL_CONFIGS.keys())}")
                sys.exit(1)
    else:
        targets = list(MODEL_CONFIGS.keys())

    # 读取 Prompt 数据
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    qids = list(prompts_data.keys())
    raw_prompts = [prompts_data[qid]["prompt"] for qid in qids]
    print(f"📥 加载了 {len(qids)} 个 Prompt，准备跑 {len(targets)} 个模型。")

    for model_name in targets:
        model_path = MODEL_CONFIGS[model_name]
        run_single_model(model_name, model_path, qids, raw_prompts)

    print(f"\n🎉 全部完成！结果在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()