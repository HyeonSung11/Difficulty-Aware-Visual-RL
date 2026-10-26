"""
import os
진단: assistant를 <think>로 prefill(continue_final_message)하면 모델이 </think> 닫고
<answer>\\boxed{}까지 내는가? (AutoProcessor 미사용 = base 테스트와 동일 구조로 CUDA fork 회피)
"""
import json, re, base64, io
from PIL import Image
from vllm import LLM, SamplingParams

IMAGE_ROOT = os.path.expandvars("${VERO_DATA}/vero-600k")
DUMP = os.path.expandvars("${VERO_DATA}/vero_runs/rollout_dumps/GSPO_qwen3vl_8b_judge_lite_4090x2_20260701_182720/1.jsonl")

def dataurl(path):
    img = Image.open(path).convert("RGB")
    b = io.BytesIO(); img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def has_pair(s, a, b):
    return s.count(a) == 1 and s.count(b) == 1 and s.index(a) < s.index(b)

samples, seen = [], set()
for ln in open(DUMP):
    o = json.loads(ln)
    inp = o["input"]; imgs = o.get("images") or []
    if not imgs or o.get("gts") in seen:
        continue
    m = re.search(r"^system\n(.*?)\nuser\n+(.*?)\nassistant\s*$", inp, re.S)
    if not m:
        continue
    samples.append((m.group(1).strip(), m.group(2).strip(), imgs[0], o.get("gts")))
    seen.add(o.get("gts"))
    if len(samples) >= 4:
        break
print(f"샘플 {len(samples)}개", flush=True)

llm = LLM(model="Qwen/Qwen3-VL-8B-Instruct", tensor_parallel_size=2, dtype="bfloat16",
          gpu_memory_utilization=0.85, enforce_eager=True, max_model_len=4096,
          limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": 262144})
sp = SamplingParams(temperature=0.6, max_tokens=1024, top_p=1.0)

PREFILL = "<think>"
n_valid = 0
for i, (sys_txt, q_txt, imgpath, gt) in enumerate(samples):
    url = dataurl(f"{IMAGE_ROOT}/{imgpath}")
    messages = [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": q_txt},
        ]},
        {"role": "assistant", "content": PREFILL},   # prefill
    ]
    out = llm.chat(messages, sampling_params=sp,
                   add_generation_prompt=False, continue_final_message=True)
    resp = out[0].outputs[0].text
    full = PREFILL + resp
    ok_t = has_pair(full, "<think>", "</think>")
    ok_a = has_pair(full, "<answer>", "</answer>")
    ok = ok_t and ok_a
    n_valid += ok
    print("=" * 70, flush=True)
    print(f"[{i}] GT={gt!r}", flush=True)
    print(f"  full(400): {full[:400]!r}", flush=True)
    print(f"  think_pair={ok_t} answer_pair={ok_a} boxed={'boxed' in full} FORMAT_OK={ok}", flush=True)
print(f"\n>>> prefill 유효 포맷: {n_valid}/{len(samples)}", flush=True)
print("DONE", flush=True)
