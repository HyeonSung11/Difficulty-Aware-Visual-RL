"""Vero 600k 데이터의 난이도(pass rate) 라벨링.

import os
각 프롬프트를 base 모델로 k번 풀려서 정답률 p = (맞은수/k)를 계산하고 tier를 붙인다.
p가 학습에 유용한 "learnable zone"(0.25~0.75)인지가 커리큘럼의 핵심.

채점은 Vero 실제 리워드 함수(vero_reward.acc_reward)를 reward_type으로 라우팅해 사용.
judge 필요 타입(llm_judge / instruction_following_llm_judge = captioning_IF)은 기본 제외
(별도 judge 서버 필요) — 규칙채점 가능한 ability부터 라벨링.

사용:
  python difficulty_label.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --per-ability 500 --k 4 --out labels.jsonl
"""
import argparse, json, os, random, sys
from collections import Counter, defaultdict

VERO_RL = os.path.expandvars("${VERO_REPO}/vero-rl")
sys.path.insert(0, VERO_RL)

DATA = os.path.expandvars("${VERO_DATA}/vero-600k/vero_600k_train.verl.jsonl")
IMG_ROOT = os.path.expandvars("${VERO_DATA}/vero-600k")
SYS_PROMPT = os.path.expandvars("${VERO_REPO}/vero-rl/examples/prompts/system_prompt_concise.txt")

JUDGE_TYPES = {"llm_judge", "instruction_following_llm_judge"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                    help="난이도 기준 policy (커리큘럼 시작점 모델). 보통 base.")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--img-root", default=IMG_ROOT)
    ap.add_argument("--per-ability", type=int, default=500,
                    help="ability별 표본 수 (0=전체). 처음엔 작게 잡고 분포 확인.")
    ap.add_argument("--k", type=int, default=4, help="프롬프트당 샘플 수")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--correct-thresh", type=float, default=0.5,
                    help="acc_reward가 이 값 이상이면 '정답'으로 카운트")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000,
                    help="한 번에 메모리에 올려 처리할 프롬프트 수 (전체 실행 시 OOM 방지)")
    ap.add_argument("--out", default=os.path.expandvars("${VERO_REPO}/curriculum/difficulty_labels.jsonl"))
    ap.add_argument("--include-judge", action="store_true",
                    help="judge 필요 타입도 포함 (judge 서버 필요, --judge-url)")
    ap.add_argument("--only-judge", action="store_true",
                    help="judge 타입만 처리 (llm_judge / instruction_following_llm_judge)")
    ap.add_argument("--judge-url", default="http://127.0.0.1:51001/v1/chat/completions")
    ap.add_argument("--normalize-bbox-1000", dest="normalize_bbox_1000", action="store_true",
                    help="grounding 좌표를 0-1000으로 간주 (config normalize_bbox_to_1000과 일치시킬 것; 기본 False)")
    return ap.parse_args()


def stratified_sample(path, per_ability, seed):
    """ability별로 per_ability개씩 층화 추출 (데이터가 ability로 정렬돼 있어도 무작위 보장)."""
    buckets = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            buckets[d.get("ability")].append(d)
    rng = random.Random(seed)
    out = []
    for ab, recs in buckets.items():
        rng.shuffle(recs)
        out.extend(recs if per_ability <= 0 else recs[:per_ability])
    rng.shuffle(out)
    return out, {k: len(v) for k, v in buckets.items()}


def reward_type_of(rec):
    ei = rec.get("extra_info") or {}
    return (ei.get("reward_type") or "").strip().lower()


def main():
    args = parse_args()
    from PIL import Image
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from vero_reward.math_verify_reward_type_boxed import acc_reward

    sys_prompt = open(SYS_PROMPT).read().strip()

    # ---- 데이터 표본 ----
    recs, ab_total = stratified_sample(args.data, args.per_ability, args.seed)
    if args.only_judge:
        recs = [r for r in recs if reward_type_of(r) in JUDGE_TYPES]
    elif not args.include_judge:
        recs = [r for r in recs if reward_type_of(r) not in JUDGE_TYPES]
    print(f"[data] ability 전체 분포: {ab_total}")
    print(f"[data] 라벨 대상 프롬프트: {len(recs)}개 (judge타입 {'포함' if args.include_judge else '제외'})")
    rt_dist = Counter(reward_type_of(r) for r in recs)
    print(f"[data] reward_type 분포: {dict(rt_dist)}")

    # ---- resume: 이미 라벨된 id 스킵 ----
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("id"))
                except Exception:
                    pass
        recs = [r for r in recs if r.get("id") not in done_ids]
        print(f"[resume] 기존 라벨 {len(done_ids)}개 발견 → 남은 {len(recs)}개만 처리")

    # ---- 모델 ----
    print(f"[model] loading {args.model} (tp={args.tp}, max_pixels={args.max_pixels}) ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                              max_pixels=args.max_pixels)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, trust_remote_code=True,
              mm_processor_kwargs={"max_pixels": args.max_pixels},
              limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, enforce_eager=True, max_model_len=8192)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, max_tokens=args.max_new)

    def build_input(r):
        q = r["prompt"][0]["content"].replace("<image>", "").strip()
        img_rel = r["images"][0] if r.get("images") else None
        if not img_rel:
            return None
        img_path = os.path.join(args.img_root, img_rel)
        if not os.path.exists(img_path):
            return None
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None
        # grounding/clicking 채점기가 좌표 정규화에 쓰는 image_sizes 주입
        # (훈련 시 rl_dataset.py가 하던 것과 동일; 안 하면 좌표채점 전부 0점)
        ei = r.setdefault("extra_info", {})
        ei["image_sizes"] = [[img.width, img.height]]
        ei["normalize_bbox_to_1000"] = args.normalize_bbox_1000
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
        ]
        prompt_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt_str, "multi_modal_data": {"image": [img]}}

    # ---- LLM judge (llm_judge / instruction_following_llm_judge 타입용) ----
    import re, urllib.request
    JUDGE_TEMPLATE = open(
        os.path.expandvars("${VERO_REPO}/vero-rl/examples/prompts/llm_judge_reference.txt")).read()

    def judge_raw(conversation, output, label):
        """reward manager와 동일: 1~10점. 실패 시 None."""
        prompt = (JUDGE_TEMPLATE.replace("{input}", str(conversation))
                  .replace("{output}", str(output)).replace("{label}", str(label)))
        body = json.dumps({"model": "qwen_vllm",
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 1024, "temperature": 0.0}).encode()
        req = urllib.request.Request(args.judge_url, data=body,
              headers={"Authorization": "Bearer qwen", "Content-Type": "application/json"})
        for _ in range(2):
            try:
                resp = json.loads(urllib.request.urlopen(req, timeout=150).read())
                txt = resp["choices"][0]["message"]["content"]
                m = re.search(r'"?SCORE"?\s*:\s*"?([0-9]+(?:\.[0-9]+)?)', txt, re.IGNORECASE)
                if not m:
                    m = re.search(r'\b(10|[1-9])(?:\.\d+)?\b', txt)
                if m:
                    return float(m.group(1))
            except Exception:
                pass
        return None

    def score_and_record(r, out):
        gt = str((r.get("reward_model") or {}).get("ground_truth", ""))
        ei = r.get("extra_info") or {}
        is_judge = reward_type_of(r) in JUDGE_TYPES
        conv = r["prompt"][0]["content"] if r.get("prompt") else ""
        n_correct = 0
        for comp in out.outputs:
            if is_judge:
                raw = judge_raw(conv, comp.text, gt)      # 1~10
                ok = (raw is not None and raw >= 6.0)      # 6점 이상 = "괜찮은 답" = pass
            else:
                try:
                    acc = acc_reward(comp.text, gt, extra_info=ei)
                except Exception:
                    acc = 0.0
                ok = acc >= args.correct_thresh
            n_correct += int(ok)
        k = len(out.outputs)
        p = n_correct / k if k else 0.0
        # tier: easy=3~4/4, medium=2/4, hard=1/4, unsolved=0/4
        tier = ("easy" if p >= 0.75 else "medium" if p >= 0.5 else
                "hard" if p > 0 else "unsolved")
        return {
            "id": r.get("id"), "ability": r.get("ability"),
            "data_source": r.get("data_source"), "reward_type": reward_type_of(r),
            "pass_rate": round(p, 3), "n_correct": n_correct, "k": k, "tier": tier,
        }

    from concurrent.futures import ThreadPoolExecutor

    def tier_of(p):
        return ("easy" if p >= 0.75 else "medium" if p >= 0.5 else "hard" if p > 0 else "unsolved")

    def score_chunk_judge(metas, outputs):
        """judge 타입: 청크의 모든 (conv,output,gt)를 병렬로 채점."""
        tasks = []  # (prompt_idx, conv, text, gt)
        per = []    # prompt별 output 개수
        for pi, (r, out) in enumerate(zip(metas, outputs)):
            gt = str((r.get("reward_model") or {}).get("ground_truth", ""))
            conv = r["prompt"][0]["content"] if r.get("prompt") else ""
            per.append(len(out.outputs))
            for comp in out.outputs:
                tasks.append((pi, conv, comp.text, gt))
        with ThreadPoolExecutor(max_workers=32) as ex:
            raws = list(ex.map(lambda t: judge_raw(t[1], t[2], t[3]), tasks))
        ncorr = [0]*len(metas)
        for (pi, *_), raw in zip(tasks, raws):
            if raw is not None and raw >= 6.0:
                ncorr[pi] += 1
        recs_out = []
        for pi, r in enumerate(metas):
            k = per[pi]; p = ncorr[pi]/k if k else 0.0
            recs_out.append({"id": r.get("id"), "ability": r.get("ability"),
                "data_source": r.get("data_source"), "reward_type": reward_type_of(r),
                "pass_rate": round(p,3), "n_correct": ncorr[pi], "k": k, "tier": tier_of(p)})
        return recs_out

    # ---- 청크 단위 처리 (메모리 안전 + 증분 저장) ----
    total = len(recs)
    tier_dist = Counter()
    byab = defaultdict(list)
    n_done = 0
    fout = open(args.out, "a")  # append (resume 대응)
    for c0 in range(0, total, args.chunk):
        chunk = recs[c0:c0 + args.chunk]
        inputs, metas = [], []
        for r in chunk:
            vi = build_input(r)
            if vi is not None:
                inputs.append(vi); metas.append(r)
        if not inputs:
            continue
        outputs = llm.generate(inputs, sp)
        judge_mode = any(reward_type_of(r) in JUDGE_TYPES for r in metas)
        if judge_mode:
            recs_out = score_chunk_judge(metas, outputs)
        else:
            recs_out = [score_and_record(r, out) for r, out in zip(metas, outputs)]
        for rec in recs_out:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tier_dist[rec["tier"]] += 1
            byab[rec["ability"]].append(rec["pass_rate"])
            n_done += 1
        fout.flush()
        print(f"[progress] {min(c0+args.chunk,total)}/{total}  tier={dict(tier_dist)}", flush=True)
    fout.close()

    print(f"\n[result] 저장: {args.out}  (이번 실행 {n_done}개)")
    print(f"[result] tier 분포: {dict(tier_dist)}")
    print("[result] ability별 평균 pass_rate:")
    for ab, ps in sorted(byab.items()):
        print(f"    {ab:16s} mean_p={sum(ps)/len(ps):.3f}  n={len(ps)}")


if __name__ == "__main__":
    main()
