# Vero 2×4090 — "학습이 안 되던" 문제 조사 기록 (2026-07-01)

작성: 새 세션에서 HANDOFF.md + RESULTS_2x4090.md를 읽고 이어서 조사한 전 과정 기록.
나중 자료 정리용. 모든 주장은 아래 증거(로그/덤프/테스트)로 뒷받침됨.

---

## TL;DR
1000스텝을 돌려도 reward가 -1.0에 고정돼 "학습이 안 되던" 문제를 두 개의 독립 버그로 분해:

| # | 버그 | 증상 | 근본 원인 | 수정 |
|---|------|------|-----------|------|
| **버그 A** | rollout = 다국어 난수 garbage | 전 스텝 reward -1.0, 응답이 캡까지 난수로 참 | vLLM sleep **level 2**가 가중치 폐기 → `load_format=dummy`라 wake 시 **랜덤 복원** | `VERL_VLLM_SLEEP_LEVEL=1` |
| **버그 B** | (A 수정 후 드러남) rollout이 `<tool_call>`/개행만, 유효형식 0/64 | reward -0.1~-0.4로 오르나 학습신호 미약 | **FSDP→vLLM base 가중치 동기화가 손상** (dummy+actor sync) | `rollout.load_format=auto` (디스크 로드로 우회) |

두 수정 모두 적용 완료. 버그 A는 검증 완료, 버그 B는 원인 확정(디스크 로드 테스트) 후 수정 적용, **재검증 대기**.

---

## 배경 (인계 시점 상태)
- 목표: Vero(arXiv 2604.04917) judged GSPO RL을 2×RTX 4090에서 재현.
- 환경: conda `verovlm` (torch 2.10+cu130, verl 0.5.0.dev, vLLM 0.17, flash-attn 2.8.3)
- policy `Qwen/Qwen3-VL-8B-Instruct` (LoRA rank32/α64, 언어모델만), judge `Qwen/Qwen3-32B-AWQ`.
- 인계 문서 결론: "파이프라인·base 재현 성공. 그러나 rollout이 garbage라 학습 안 됨." + "응답 예산 부족(하드웨어 천장)" 추정.

---

## 조사 타임라인 & 증거

### 1. 문제 재현 확인
- `rollout_360.jsonl`(= evidence_dumps/A_garbage_era_step360.jsonl) 및 dump 141~440 전수 스캔:
  전 스텝 output이 다국어 난수 토큰(`.Dialog很多时候 displayName spacer...`), score/format 전부 0.
- 학습 로그(`_logs/judge_concise_dump_20260630_122038.log`): step 439까지도 `critic/rewards/mean:-1.0`,
  `response_length/clip_ratio:1.0` (응답이 캡에 꽉 참).

### 2. 버그 A 진단 — 증거 사슬
1. **LoRA는 무죄**: 저장된 어댑터(`global_step_90/.../lora_adapter/adapter_model.safetensors`) 조사 →
   lora_B mean|.|≈**7.7e-5**, lora_A≈7.3e-3, NaN 0. ΔW=(α/r)·B@A≈2.0×~1e-6 = near-identity.
   → 이렇게 작은 LoRA는 절대 garbage를 못 만든다.
2. **hijack 재작성도 무죄**: `verl/utils/vllm/utils.py`의 `from_lora_tensors` 호출이 vLLM 0.17
   실제 시그니처/구현과 정확히 일치. 스케일 2.0은 α64/r32의 정상값.
3. **진짜 원인 = sleep level 2 + dummy**:
   - `verl/third_party/vllm/__init__.py`: vLLM≥0.8.5 → `VLLM_SLEEP_LEVEL=2`.
   - vLLM `v1/worker/gpu_worker.py` `sleep()`: `offload_tags=("weights",) if level==1 else tuple()`
     → **level 2는 가중치를 offload하지 않고 폐기**.
   - `fsdp_vllm.py __exit__`가 매 스텝 sleep(level 2) → 가중치 폐기. wake 시 `load_format=dummy`
     소스에서 재로드 = **랜덤**. `base_sync_done`은 첫 스텝 후 True로 latch돼 base 재동기화 안 함.
   - ⇒ 매 wake마다 랜덤 base + near-identity LoRA = 난수 토큰. "step1 정상(방금 sync) → step2+ garbage" 일치.
   - **부수 정정**: RESULTS §3의 "응답 예산 부족" 결론은 오진. clip_ratio=1.0은 CoT가 길어서가 아니라
     난수가 `\boxed{}` 없이 max까지 채운 것. 하드웨어 천장이 아니라 이 버그가 원인.

### 3. 버그 A 수정 & 검증
- 수정: `verl/third_party/vllm/__init__.py`에 env override 추가 →
  `run_gspo_qwen3vl_judge_lite.sh`에 `export VERL_VLLM_SLEEP_LEVEL=1`.
  (level 1 = 가중치를 CPU RAM에 보존, GPU 비용 0, host RAM +~16GB. 노드 125GB라 안전.)
- garbage 런(step 440) 중단 → fix 적용해 재시작.
- **검증 결과 (evidence_dumps/B_level1fix_era_step*.jsonl, 로그 `..._174219.log`)**:
  - 난수 유니코드 **사라짐**(ascii_frac 0.85→1.00), reward **-1.0→-0.125**,
    response_length 1536→234, clip_ratio 1.0→0.125. ⇒ 버그 A 확실히 해결.

### 4. 버그 B 발견 (A 아래 가려져 있던 문제)
- 그러나 재시작 후 step1~8의 **64개 rollout 중 `<think>`/`<answer>`/`\boxed{}` 0개**.
  49개가 개행/`<tool_call>` 반복, 나머지도 형식 미준수. 입력 프롬프트/이미지는 정상(gts·이미지 존재 확인).
- 논리: 난수 가중치→난수 토큰이지만, **올바른 가중치+손상된 입력→개행/특수토큰 반복**. 지금은 후자 패턴.

### 5. 버그 B 결정적 진단 — base를 디스크에서 로드
- 테스트: `diagnostics/vllm_base_test.py` — base Qwen3-VL-8B를 vLLM에 **디스크에서 직접 로드**(dummy 아님),
  rollout과 동일한 프롬프트+이미지+샘플링(temp 0.6)으로 생성.
- **결과 (diagnostics/base_from_disk_test_result.out)**: 3/3 샘플 정답까지 정확.
  - Q "top 10% 부 비율?" → `\boxed{77.1}` (정답 77.1) ✅
  - Q "몇 명?" → `\boxed{2}` (정답 2) ✅
- ⇒ 모델·vLLM·멀티모달·프롬프트 전부 정상. **범인 = FSDP→vLLM base 가중치 동기화** 확정.
  (디스크 로드는 정상, dummy+동기화한 가중치만 degenerate.)

### 6. 버그 B 수정 (적용, 재검증 대기)
- 수정: `config/gspo_qwen3vl_judge_lite.yaml`의 rollout에 `load_format: auto` 추가.
  - base를 디스크에서 로드(정상 검증됨) → 손상된 동기화 경로를 **우회**.
  - base는 frozen(LoRA)이라 디스크 base == 학습 base. LoRA만 add_lora로 hot-swap(정상 경로).
  - `base_sync_done`이 init부터 True → 손상된 base sync 미호출.
  - sleep level-1과 결합 → wake마다 디스크 재로드 없이 base 보존.

---

## 적용된 코드 변경 (파일별)
1. `vero-rl/verl/third_party/vllm/__init__.py` — `VERL_VLLM_SLEEP_LEVEL` env override.
2. `vero-rl/examples/model_runs/run_gspo_qwen3vl_judge_lite.sh` — `export VERL_VLLM_SLEEP_LEVEL=1`.
3. `vero-rl/examples/model_runs/config/gspo_qwen3vl_judge_lite.yaml` — `rollout.load_format: auto`.

## 원본 로그 위치 (raw)
- 학습 로그: `vero-rl/_logs/judge_concise_dump_*.log`
  - `20260630_122038` = 버그 A garbage 런(step 440에서 중단)
  - `20260701_174219` = 버그 A 수정 후 런(버그 B 드러남, step ~10에서 중단)
- vLLM judge 서버 로그: `vero-rl/examples/model_runs/vllm_logs/vllm_judge_51001_*.txt`
- rollout 덤프(원본, **스텝번호로 덮어써짐 주의**): `${VERO_DATA}/vero_runs/rollout_dumps/*.jsonl`
  - 대표 증거는 `evidence_dumps/`에 보존.
- 체크포인트: `${VERO_DATA}/vero_runs/checkpoints/verl/GSPO_qwen3vl_8b_judge_lite_4090x2_*/`

### 7. 이슈 C — 버그 B 수정 후 드러남 (보상/형식 불일치)
- load_format=auto 재시작 후: 모델이 유효한 `<answer>\boxed{}`를 생성(step1 boxed 6/8),
  일부는 정답("Groundhog Day", `\boxed{2000}`). 그러나 **score 전부 0.0, reward 0, grad_norm 0**.
- 원인: `vero_reward/math_verify_reward_type_boxed.py`
  - `format_reward`: `<think></think>` 완전쌍 없으면 0 반환(line 1031).
  - `compute_score`: `if formatting==0.0: accuracy=0.0` (line 1127) → **포맷 실패 시 정답도 0점**.
  - 집계: 재시작 후 80개 rollout에서 `<think></think>` **0%**, `<answer>` 63%, boxed 51%,
    `<tool_call>` 62%. 즉 Instruct(non-thinking) 모델이 `<think>` 대신 `<tool_call>`을 냄.
- 진단(diagnostics/prefill_test_finding.md): `<think>` prefill해도 모델이 `</think>` 대신
  **`</answer>`로 reasoning을 닫음** → prefill 단독으론 format 0/4. 단 끝의 `<answer>\boxed{답}</answer>`와
  정답 내용은 정확.

### 8. 이슈 C 수정 (prefill + 태그 정규화, 논문 충실 방식)
- **프롬프트 prefill**: `verl/utils/dataset/rl_dataset.py`에 `RESPONSE_PREFILL` env(`VERL_RESPONSE_PREFILL`)
  추가 → `__getitem__`의 프롬프트 끝에 `<think>` 붙임(응답이 `<think>`로 시작하도록 강제).
  overlong-filter doc2len 경로엔 안 넣어 **필터 캐시 무효화(2h 재필터) 회피**.
- **보상 정규화**: `verl/workers/reward_manager/vero_vllm_judge.py`에 `_restore_prefill_and_normalize`
  추가 → 채점 전 응답 앞에 `<think>` 복원 + `<answer>` 없이 앞선 첫 `</answer>`를 `</think>`로 교정
  → `<think>추론</think><answer>\boxed{답}</answer>` 논문 형식 복원.
- **단위 검증(GPU 불필요)**: 진단 응답 3개로 확인 → `\boxed{2000}`(gt2000)·`\boxed{A}`(gtA)가
  format=1.0, accuracy=1.0, **score=1.0**. (answer 태그 없는 bare boxed만 0 = 정당한 실패.)
- 실행: `run_gspo_qwen3vl_judge_lite.sh`에 `export VERL_RESPONSE_PREFILL='<think>'` 추가.

## 적용된 수정 (누적, 4개 파일 + 3개 env/config)
1. `verl/third_party/vllm/__init__.py` + `VERL_VLLM_SLEEP_LEVEL=1` (버그 A)
2. `config: rollout.load_format=auto` (버그 B)
3. `verl/utils/dataset/rl_dataset.py` + `verl/workers/reward_manager/vero_vllm_judge.py`
   + `VERL_RESPONSE_PREFILL='<think>'` (이슈 C)
4. `run 스크립트: trainer.rollout_data_dir` 런별 분리 (로그 보존)

## 다음 할 일
1. 세 수정 모두 적용해 재시작 → 첫 몇 스텝에서 정답이 score>0 받고 reward가 오르는지 확인.
2. reward가 상승하면 학습 곡선 모니터링(`plot_reward.py`).
3. (여력되면) FSDP→vLLM base 동기화 근본 규명 — 지금은 디스크 로드로 우회.
