"""
import os
결정적 진단: base Qwen3-VL-8B-Instruct를 vLLM에 디스크에서 직접 로드(dummy 동기화 아님)해
rollout과 동일한 프롬프트+이미지+샘플링으로 생성.
- 정상 답(\boxed{}) 나오면 -> vLLM/멀티모달/프롬프트는 정상, 범인은 FSDP->vLLM 가중치 동기화.
- 그래도 junk면 -> vLLM 멀티모달 처리/프롬프트 자체 문제.
"""
import json, re, base64, io
from PIL import Image
from vllm import LLM, SamplingParams

IMAGE_ROOT = os.path.expandvars("${VERO_DATA}/vero-600k")
DUMP = os.path.expandvars("${VERO_DATA}/vero_runs/rollout_dumps/2.jsonl")

def img_to_dataurl(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

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
    if len(samples) >= 3:
        break
print(f"추출 샘플 수: {len(samples)}", flush=True)

llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={"max_pixels": 262144},
)
sp = SamplingParams(temperature=0.6, max_tokens=1024, top_p=1.0)

for i, (sys_txt, q_txt, imgpath, gt) in enumerate(samples):
    url = img_to_dataurl(f"{IMAGE_ROOT}/{imgpath}")
    messages = [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": q_txt},
        ]},
    ]
    outs = llm.chat(messages, sampling_params=sp)
    text = outs[0].outputs[0].text
    print("=" * 70, flush=True)
    print(f"[sample {i}] Q: {q_txt[:90]}", flush=True)
    print(f"  GT: {gt}", flush=True)
    print(f"  OUTPUT ({len(text)} chars):", flush=True)
    print("  " + text[:800].replace(chr(10), "\n  "), flush=True)
    ok = ("boxed" in text) or ("<answer>" in text)
    print(f"  >>> 유효형식(boxed/answer): {ok}", flush=True)
print("DONE", flush=True)
