"""VisualProbe samples를 judge 서버로 직접 채점 (이미지 불필요, 텍스트만).
import os
생성은 이미 끝났고 samples에 question/reference/prediction이 들어있음."""
import json, glob, os, urllib.request, re

_VP_DIR = os.environ.get("VP_RESULTS_DIR",
    os.path.expandvars("${VERO_REPO}/investigation_2x4090/eval_results_fair"))
_cands = sorted(glob.glob(os.path.join(_VP_DIR, "**",
    "*samples_visual_probe_easy_reasoning.jsonl"), recursive=True))
if not _cands:
    raise SystemExit(f"VisualProbe samples 파일 없음: {_VP_DIR}")
SAMPLES = _cands[-1]
print(f"채점 대상: {SAMPLES}", flush=True)
JUDGE_URL = "http://127.0.0.1:51001/v1/chat/completions"
PROMPT = """You are given a visual search question, the reference answer, and a model answer.
Decide if the model answer correctly addresses the question. Treat capitalization,
punctuation, and articles as irrelevant. Accept close paraphrases and synonyms.
Mark incorrect if they refer to the wrong object/attribute, are unrelated, or miss required specifics.

Question: {q}
Reference answer: {r}
Model answer: {p}

Respond with a JSON object only: {{"is_equivalent": true/false}}."""

def judge(q, r, p):
    body = json.dumps({"model": "qwen_vllm",
        "messages": [{"role": "user", "content": PROMPT.format(q=q, r=r, p=p)}],
        "max_tokens": 200, "temperature": 0.0}).encode()
    req = urllib.request.Request(JUDGE_URL, data=body,
        headers={"Authorization": "Bearer qwen", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
            txt = resp["choices"][0]["message"]["content"]
            m = re.search(r'\{[^{}]*\}', txt)
            if m:
                d = json.loads(m.group(0))
                v = d.get("is_equivalent")
                if isinstance(v, bool): return v
                if isinstance(v, str): return v.strip().lower().startswith(("y", "t"))
            low = txt.lower()
            if "true" in low: return True
            if "false" in low: return False
        except Exception as e:
            last = str(e)
    return False

correct = 0; n = 0
for ln in open(SAMPLES):
    o = json.loads(ln); n += 1
    ja = o.get("judge_accuracy", {})
    q = ja.get("question", o.get("input", ""))
    r = ja.get("reference", str(o.get("target", "")))
    p = ja.get("prediction") or ja.get("raw_prediction", "")
    if not p:
        fr = o.get("filtered_resps", [""])
        p = fr[0] if isinstance(fr, list) else str(fr)
    ok = judge(q, r, p)
    correct += int(ok)
    if n % 30 == 0: print(f"  {n}  acc={correct/n:.3f}", flush=True)
print(f"\n=== VisualProbe judge_accuracy = {correct/n:.4f}  ({correct}/{n}) ===")
