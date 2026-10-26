# Vero 2×RTX 4090 재현 · 제약 환경 학습 보고서

## 0. 개요
- **납)
- 도메인 균형 샘플링 끔, 검증 끔목표**: Vero(*An Open RL Recipe for General Visual Reasoning*, arXiv 2604.04917)를
  논문의 8×A100/H100 대신 **2×RTX 4090(각 24GB)** 에서 재현·확장.
- **환경**: conda `verovlm` (torch 2.10+cu130, verl 0.5.0.dev, vLLM 0.17, flash-attn 2.8.3)
- **모델**: policy `Qwen/Qwen3-VL-8B-Instruct`, judge `Qwen/Qwen3-32B-AWQ`
- **데이터**: Vero-600k (train 60만, 이미지 포함)

---

## 1. 성과 요약

### 1-1. 파이프라인 전체 검증 ✅
judged GSPO RL 루프가 2×4090에서 끝까지 동작:
**rollout 생성 → log-prob → LLM-judge 보상 → LoRA 업데이트 → 어댑터를 vLLM에 재동기화**

핵심 적용:
- **LoRA** (rank 32, α 64, 언어모델만) — full-param 대신
- **bf16** actor(`model_dtype`), 오프로딩 끔
- **AWQ 4bit judge** + vLLM **sleep level 2** (대기 중 GPU·호스트 RAM 모두 반

### 1-2. base 모델 eval = 논문 재현 성공 ⭐
`vero-eval`(lmms_eval 포크)로 **base** Qwen3-VL-8B-Instruct 평가:

| 벤치마크 | 우리 측정 | 논문(Q3VL8B-Ins) |
|---|---|---|
| CountBenchQA (`countbenchqa_qwen3_zs`) | **89.0%** | **88.8%** |

> **함정 발견**: base 모델은 **`_qwen3_zs`** 변형("숫자로만 답하라")으로 평가해야 함.
> 추론 학습 모델용 `_reasoning`을 base에 쓰면 78.4%로 잘못 나옴(과소평가).
> 올바른 변형으로 논문 88.8 재현.

---

## 2. 버그 수정 전체 로그 (verl + vLLM 0.17 + Qwen3-VL 호환)

### 단계 1 — 스모크 테스트 (judge OFF, 파이프라인 검증)
| # | 에러 | 원인 | 수정 |
|---|---|---|---|
| ① | `NaiveRewardManager ... 'format_score'` | shared의 reward_kwargs가 naive로 splat | `'~reward_model.reward_kwargs'` 오버라이드 |
| ② | 필터 캐시 미저장(매번 2h) | `HF_DATASETS_CACHE` 미설정 | 스크립트에 export |
| ③ | (LoRA 도입) | VRAM 대응 | `lora_rank 32` 등 |
| ④ | `No per_batch_domain_weights domains` | 데이터 `ability` 라벨 불일치 | `per_batch_domain_weights: null` |
| ⑤ | chunked prefill 토큰 한도 | `max_num_batched_tokens < max_model_len` | batched_tokens ↑ |
| ⑥ | KV 캐시 `No available memory` | `gpu_memory_utilization 0.4` 낮음 | 0.6 |
| ⑦ | 호스트 RAM OOM (워커당 ~50GB) | param/optimizer offload + fp32 로드 | offload 끔 |
| ⑧ | GPU OOM (actor fp32 상주) | **actor가 model_dtype 미설정 시 fp32** | **`model_dtype: bfloat16`** (⑦⑧ 공통 근본) |
| ⑨ | `KeyError '...qkv.base_layer.weight'` | all-linear이 비전 타워에도 LoRA | `exclude_modules: ".*visual.*"` |
| ⑩ | deepstack `IndexError [1,6864] vs [1,3432]` | Qwen3-VL deepstack ↔ Ulysses SP 비호환 | `ulysses_sequence_parallel_size: 1` |
| ⑪ | LoRA hijack ↔ vLLM 0.17 (3겹) | verl 몽키패치가 구버전 API | hijack을 0.17 API로 재작성 |

→ **스모크 10스텝 통과 = 파이프라인 검증 완료**

### 단계 2 — 실제 judged 학습 (judge ON)
| # | 에러 | 원인 | 수정 |
|---|---|---|---|
| ⑫ | `Image features and image tokens do not match: 313 vs 312` | vLLM↔HF 이미지 처리 반올림 차 | `_get_input_embeds` off-by-one 정렬 |
| ⑬ | deepstack 같은 불일치 전파 | main만 정렬, deepstack 피처 누락 | deepstack 피처도 정렬 |
| ⑭ | GPU OOM (compute_log_prob, 1.46GB) | rollout 12GB 풀 + actor → logits 자리 없음 | (util 조정 제로섬 확인) |
| ⑮ | util 조정 무효(제로섬) | 0.45→rollout KV 실패 / expandable↔judge 충돌 | **max_prompt 2048→1024** (시퀀스 축소) |

→ **1000스텝 완주**(단, 학습은 안 됨 — 3장). max_prompt 축소로 train 60만→27만 재필터(1회 2h).

---

## 3. 하드웨어 천장 — 왜 "학습"은 안 됐나 ⚠️
1000스텝을 크래시 없이 완주했지만 **보상이 끝까지 -1.0 바닥에 고정**(약 91% 스텝, advantage 0).

**진단(추측 아님)**: `response_length/clip_ratio == 1.0`(응답이 캡에 꽉 참)과
`reward == -1.0`이 **완전히 일치**. 모델이 긴 CoT를 응답 예산 안에 못 끝내 `\boxed{}` 답을
못 냄 → 무효 → -1.0 → 학습 신호 없음.

| 항목 | 원본 Vero | 우리(2×4090) |
|---|---|---|
| max_prompt | 12,288 | 1,024 |
| max_response | **16,384** | **1,024** |
| max_pixels | 9,437,184 | 1,048,576 |
| rollout n | 8 | 2 |
| batch | 256 | 4 |
| 하드웨어 | 8×A100/H100 | 2×4090 |

응답 예산이 레시피의 약 1/16. temperature 1.0→0.6도 생성 길이를 못 줄임(모델이 온도와
무관하게 길게 추론). 즉 2×4090에선 응답 길이가 근본적으로 부족해 충실한 judged-RL 학습이 불가.

---

## 4. 결론
- **파이프라인 + base 재현: 성공.** 컨슈머 GPU에서 Vero식 judged GSPO를 돌리는 데 필요한
  모든 것이 동작·문서화됐고 base eval이 논문과 일치.
- **충실한 RL 학습: VRAM 한계로 막힘.** 의미 있는 학습엔 큰 응답 예산(16k)이 필요 → 8×A100급.
- **시도 중(B)**: `max_pixels`를 낮춰 이미지 위주 프롬프트를 줄이고 그 예산을 응답으로 재배분.

---

## 부록 — 2×4090에서 돌아가는 config
- `examples/model_runs/config/gspo_qwen3vl_judge_lite.yaml`
- `examples/model_runs/run_gspo_qwen3vl_judge_lite.sh`
- (AWQ judge, util 0.5, max_num_batched_tokens 2048, token 예산 2048, save_freq 10, 검증 끔)
