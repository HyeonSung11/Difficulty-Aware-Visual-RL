# Vero 공정판(FAIR) 평가 매니페스트 — "우리가 쓴 그대로"

> 목적: 향후 개선 모델을 **동일 프로토콜로 한 번에** 평가하기 위한 단일 기준.
> 러너: `run_full_eval.sh <MERGED_MODEL_PATH>` (LoRA면 먼저 merge_lora.py로 병합).

## 전 태스크 공통 (채점에 영향 → 절대 고정)

| 파라미터 | 값 | 비고 |
|---|---|---|
| 시스템 프롬프트 | `examples/prompts/system_prompt_concise.txt` | 공정판 핵심(‑> `\boxed{}` 추출) |
| `max_pixels` | **262144** (~1024²) | 학습과 일치, 전 태스크 동일 |
| `gen_kwargs max_new_tokens` | **512** | 전 태스크 동일 (rambling 방지) |
| `temperature` | 태스크 yaml 기본(0.6) | |
| `enforce_eager` | True | |

## 메모리/인프라 (출력 불변 — 태스크 요구량 이상이면 결과 동일)

| 그룹 | TP | `max_model_len` | GPU 배치 | judge |
|---|---|---|---|---|
| A. no-judge | **2** | 32768 | GPU0+1 (max KV) | 불필요 |
| B. judge | 1 | 18432 | eval=GPU0, judge=GPU1 | Qwen3-32B judge 서버 |

> `max_model_len`은 KV 버퍼 크기일 뿐 — 실제 (프롬프트+응답)보다 크기만 하면 출력 동일.
> VisualProbe는 프롬프트가 16405토큰이라 **32768 필수**(그래서 그룹 A/TP2에서 생성).

## 태스크별 명세 (30개)

### 그룹 A — no-judge (TP=2, max_model_len 32768) — 17개
`chartqa_reasoning, chartqa_pro_reasoning, infovqa_val_reasoning, evochart_reasoning,`
`countbenchqa_reasoning, countqa_reasoning, mme_realworld_lite_reasoning, vstar_bench_reasoning,`
`screenspot_point_in_box_reasoning, visual_probe_easy_reasoning, mathvista_testmini_reasoning,`
`cv_bench_reasoning, blink_multi_view_reasoning, erqa_reasoning, embspatial_reasoning,`
`game_qa_lite_reasoning, realworldqa_reasoning`

### 그룹 B — judge 필요 (TP=1, judge 서버 GPU1) — 9개
`charxiv_reasoning, chartmuseum_reasoning, mmvetv2_group_img_reasoning, simplevqa_en_reasoning,`
`mmmu_pro_reasoning, mathvision_test_reasoning, mm_mt_bench_reasoning, mia_bench_reasoning, mmifeval_reasoning`

## 특수 처리 (반드시 그대로 재현)

| 태스크 | 특수 처리 | 이유 |
|---|---|---|
| **screenspot_point_in_box** | 코드 수정 상시 적용: `utils_point.py`(이미지크기 영속화) + `reasoning.yaml`(rel1000 좌표) | 안 하면 0.2 (좌표계·크기 버그) |
| **chartmuseum** | `--limit 514` | doc 514(2540×3117)가 vLLM Qwen3-VL masked-scatter 유발 → n=514 부분측정 |
| **visual_probe** | 그룹A에서 생성만, judge는 **오프라인** `score_visualprobe.py` | TP2엔 judge 없음 + 16405토큰 |
| **captioning 3종** (mm_mt/mia/mmifeval) | judge: **non-thinking 템플릿**(`chat_template_no_think.jinja`) + judge `max-model-len ≥ 8192` | judge가 `<think>` 뱉으면 파싱 실패, max_tokens>len이면 400 |

## judge 서버 설정 (그룹 B)
- 모델: `Qwen/Qwen3-32B-AWQ` (GPU1)
- 템플릿: `chat_template_no_think.jinja` (기본값)
- `max-model-len`: 8192
- 엔드포인트: `http://127.0.0.1:51001/v1/chat/completions`
- 환경변수: `OPENAI_API_URL`, `OPENAI_API_KEY=qwen`, `JUDGE_MODEL_PATH=qwen_vllm`

## 측정 불가 3개 (개선판에서도 동일 제외)
`AerialVG`(HF gated) · `ScreenSpot-Pro`(데이터 없음) · `FVQA`(라이브러리 비호환)

## 사용법
```bash
# 1) (LoRA인 경우) 어댑터 병합
python investigation_2x4090/merge_lora.py --adapter <ckpt> --out <merged_path>
# 2) 전체 평가 한 번에 (그룹 A→B 자동)
bash eval_harness/run_full_eval.sh <merged_path>
# 3) 결과 취합
#    - 그룹 A/B: lmms_eval 결과 jsonl (eval_results_fair_<model>/)
#    - visual_probe: score_visualprobe.py 자동 실행
#    - 표 생성: investigation_2x4090 의 표 스크립트에 점수 입력
```
