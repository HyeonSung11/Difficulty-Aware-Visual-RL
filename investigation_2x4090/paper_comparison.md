# Vero 2×4090 재현 모델 vs 논문 결과 비교

- **우리 모델**: Qwen3-VL-8B-Instruct + LoRA(rank32), GSPO judged-RL, **terse CoT**, 2×RTX 4090, **4000 step**
  (batch=4, n=4 — 논문 batch=256, n=8의 1/64~1/128 스케일)
- **논문 모델**: Vero-Qwen3I-8B (동일 base, full-scale RL). 출처: arXiv 2604.04917 Table 3.
- 모든 수치는 `_reasoning` 변형(vero-eval)으로 측정. 논문 base/Vero는 Table 3 값.

## 벤치마크 비교

| 벤치마크 | 논문 base | 논문 Vero | **우리(2×4090)** | 우리 vs base | 우리 vs Vero |
|---|---|---|---|---|---|
| CV-Bench | 85.5 | 87.9 | **86.8** | +1.3 ✅ | −1.1 |
| V*Bench | 82.2 | 89.5 | **85.9** | +3.7 ✅ | −3.6 |
| MathVista (testmini) | 77.2 | 78.7 | **71.2** | −6.0 | −7.5 |
| CountBenchQA | 88.8 | 90.4 | **77.2** | −11.6 | −13.2 |
| ChartQA | 89.6 | 91.6 | **42.9** ⚠️ | −46.7 | −48.7 |
| RealWorldQA | 71.5 | 73.3 | **45.1** ⚠️ | −26.4 | −28.2 |
| **6개 평균** | **82.5** | **85.2** | **68.2** | **−14.3** | **−17.0** |
| 급락2개 제외 4개 평균 | 83.4 | 86.6 | **80.3** | −3.1 | −6.3 |

## 해석

**혼합 결과 — 균일한 실패가 아님:**
- ✅ **CV-Bench(86.8)·V\*Bench(85.9)**: base 동급/이상. V\*Bench는 base보다 +3.7. 객관식이라 추출이 안정적.
- ➖ **MathVista(71.2)·CountBenchQA(77.2)**: base보다 6~12점 낮음(정체~약간 손상).
- ⚠️ **ChartQA(42.9)·RealWorldQA(45.1) 급락**: **대부분 프롬프트/추출 불일치 아티팩트**.
  - 우리 모델은 학습 시 Vero 시스템프롬프트(`<think><answer>\boxed{}`) + `<think>` prefill로 학습됨.
  - 표준 `_reasoning` 태스크는 그 시스템프롬프트를 안 줌 → 모델이 **장황하게 추론하다 잘리고 `\boxed{}`를 거의 안 냄**(ChartQA 2500개 중 58개만) → relaxed match 추출 실패.
  - CV/V\*처럼 객관식은 끝에 선택지가 나와 추출 성공 → 정상. 즉 정밀 추출이 필요한 ChartQA/RealWorldQA만 급락.

**정직한 결론:**
- 우리 terse 2×4090 모델은 **논문의 정확도 향상을 재현하지 못함**(6개 평균 68.2 vs 논문 base 82.5, Vero 85.2).
- 원인 2가지: (1) **스케일 부족**(batch 1/64, 데이터 1/1000 → 학습 중 accuracy 정체·grad=0 50%+), (2) **terse+좁은 학습 + 평가 프롬프트 불일치**(논문이 경고한 regime).
- 단, 급락 2개는 진짜 능력 상실이라기보다 평가 조건 불일치 성격이 큼 → 학습 프롬프트로 재평가하면 덜 나쁠 것.

## 참고: 논문의 terse 경고 (Figure 9)
`64.8→26.8`은 전체 평균이 아니라 **Captioning & Instruction Following** 카테고리 평균. 구조화/terse 출력만으로 학습하면 **대화·지시따르기 능력이 붕괴**(64.8→26.8), open-ended 데이터+LLM-judge 넣어야 70.6으로 회복. 우리는 정확히 이 terse-only regime에 해당.

## 원자료
- 우리 eval 로그: `vero-rl/_logs/eval_trained_multi_20260706_104853.log`, `eval_trained_countbench_20260706_092830.log`
- 샘플: `investigation_2x4090/eval_results/*_samples_*.jsonl`
- 논문 수치: arXiv 2604.04917 Table 3 / Figure 9 (WebFetch)
