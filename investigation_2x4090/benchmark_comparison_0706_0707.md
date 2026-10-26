# 7/6~7/7 — 완주 모델의 논문 벤치마크 비교 (과정 + 결과표)

## 큰 그림: 왜 벤치마크 비교로 갔나
4000 step 학습이 끝난 이상 핵심 질문은 **"이 2×4090 축소 학습이 실제로 논문을 재현하는가?"** 이다.
학습 중 본 accuracy(~0.45)는 **학습데이터 상 정답률**일 뿐이고, 진짜 성능은 **held-out 벤치마크 +
논문과의 정량 비교**로만 알 수 있다. 그래서 7/6~7/7은 전부 "완주 모델을 평가체계에 올려 논문
Table 3와 1:1로 비교"하는 작업이었다.

---

## 진행 과정 (왜 / 어떻게 / 결과)

### [7/6] 완주 확인 → 첫 비교 시작
1. **완주 확인**: "GPU 멈춤" = 크래시 아니라 **4000/4000 완주**(7/4 종료). 최종 체크포인트 + 종료
   validation(acc 0.599, format 0.993).
2. **LoRA 병합**: eval 프레임워크(vero-eval)는 단일 모델 경로만 받으므로, base+LoRA 어댑터를
   **병합해 17.5GB 단독 모델** 생성 (그래야 평가 가능).
3. **CountBenchQA 먼저**: 이전 세션에서 base 기준값(89%)을 가진 유일한 벤치마크라 바로 비교 가능.
   → 우리 **77.2%** ≈ base reasoning(78.4) = **향상 없음** (첫 신호).
4. **논문 수치 WebFetch**: 추측 금지 → 논문에서 직접. Table 3(벤치마크), Figure 9(terse) 확보.
   - 핵심: **terse 64.8→26.8은 Captioning 카테고리**(전체 아님)의 붕괴 경고. Vero 전체 +5.5(60.5→66.0).
5. **첫 비교표(6개)**: 하나로는 판단 불가 → 도메인별 5개 더(ChartQA·MathVista·V\*Bench·RealWorldQA·
   CV-Bench, `_reasoning`·judge불필요라 간단) 추가. → 우리 평균 **68.2** vs base 82.5 / Vero 85.2.
   혼합: CV·V\*는 base 동급↑, MathVista·CountBench 낮음, **ChartQA·RealWorldQA 급락**
   (나중에 "프롬프트 불일치 아티팩트"로 판명).

### [7/7] 전체 Table 3(~28개) 재현으로 확대
6개는 부분 비교라 "골라 본 것" 비판 가능 → **전체를 돌려 완결된 비교표**를 만들기로(옵션 B).
1. **출처 확정**: 벤치마크=Table 3, terse=Figure 9.
2. **28개 전수조사**: LLM judge 필요한 태스크는 32B 서버를 같이 띄워야 하므로 미리 분류 →
   **judge 불필요 ~16개 / 필요 ~11개**(Captioning 3개는 judge 필요).
3. **논문 전 수치 확보**: WebFetch로 Table 3 전체(6도메인 27개 base/Vero + 카테고리평균 + 전체
   60.5→66.0) → `paper_table3_reference.md` 저장.
4. **no-judge 배치 → 연쇄 데이터셋 문제 → 사전점검**:
   - 1차: **AerialVG(HF gated)** 로 배치 통째 실패 (lmms_eval이 모든 데이터셋을 먼저 로드 →
     하나만 깨져도 전체 죽음).
   - 2차: **ScreenSpot-Pro(데이터 파일 없음)** 로 또 실패.
   - 교훈: 매 실패마다 8B 재로드는 낭비 → **전 데이터셋을 lmms_eval 로더로 사전점검**해 깨진 것
     전부 걸러냄 → **깨끗한 10개 배치 실행 성공**.

---

## 결과표 (논문 Table 3 vs 우리, `_reasoning` 변형)

값 = 정확도(%). 우리(표준) = 시스템프롬프트 없이 평가한 표준 프로토콜.
⚠️ = 프롬프트/추출 불일치 아티팩트(우리 모델이 학습 프롬프트 없이 산문형으로 답해 추출 실패).

| 도메인 | 벤치마크 | 논문 base | 논문 Vero | 우리(표준) | 비고 |
|---|---|---|---|---|---|
| Chart&OCR | ChartQA | 89.6 | 91.6 | 42.9 | ⚠️ 추출 |
| Chart&OCR | ChartQA-Pro | 44.3 | 60.2 | 35.8 | |
| Chart&OCR | InfoVQA | 83.1 | 87.8 | 3.2 | ⚠️⚠️ (공정판 79.2) |
| Chart&OCR | EvoChart | 64.0 | 75.7 | 45.3 | |
| Grounding·Count | CountBenchQA | 88.8 | 90.4 | 77.2 | |
| Grounding·Count | CountQA | 28.5 | 33.9 | 25.1 | |
| Grounding·Count | MME-RealWorld | 47.1 | 57.8 | 49.6 | ✅ base↑ |
| Grounding·Count | V*Bench | 82.2 | 89.5 | 85.9 | ✅ base↑ |
| Grounding·Count | ScreenSpot | 86.6 | 93.6 | 0.2 | ⚠️ 좌표(공정판도 2.0) |
| STEM | MathVista | 77.2 | 78.7 | 71.2 | |
| Spatial&Action | CV-Bench | 85.5 | 87.9 | 86.8 | ✅ base↑ |
| Spatial&Action | Blink | 69.1 | 68.7 | 32.3 | ⚠️ |
| Spatial&Action | ERQA | 45.8 | 43.2 | 43.8 | ✅ 동급 |
| Spatial&Action | EmbSpatial | 78.5 | 79.2 | 77.6 | ✅ 동급 |
| Spatial&Action | GameQA | 34.0 | 52.3 | 26.8 | |
| Knowledge | RealWorldQA | 71.5 | 73.3 | 45.1 | ⚠️ |

**미측정/불가:** AerialVG(gated), ScreenSpot-Pro(데이터 없음). judge 필요 11개(CharXiv·ChartMuseum·
VisualProbe·MMMU-Pro·MathVision·SimpleVQA·FVQA·MM-Vet·MM-MTBench·MIABench·MMIFEval)는 별도 진행 중.

## 요약 (지금까지 16개 기준)
- 우리(표준) 평균 ≈ **50 초반**, 논문 base 68 / Vero 72 수준 (아티팩트 다수 포함 시).
- **아티팩트 제외**하면 base 근접(예: CV·V\*·EmbSpatial·ERQA·MME는 base 동급/이상).
- 핵심 결론: **논문의 정확도 향상은 재현 못 함**(스케일 1/128·LoRA·저해상도·terse).
  단 붕괴로 보이는 상당수는 **평가 프롬프트 불일치 아티팩트**(공정판에서 InfoVQA 3.2→79.2로 검증).

## 다음
- 공정판(시스템프롬프트 주입) 전체 재측정 진행 중.
- Captioning 3개(judge 필요) → 논문 terse(26.8)/full(83.8)과 직접 비교 예정.
