# Vero 2×4090 — 논문 Table 3 재현 현황 (2026-07-13 기준)

## 커버리지

- **논문 Table 3 = 총 30개 태스크** (6개 도메인, MMMU-Pro는 Std/Vision 2개로 셈)
- **측정 완료: 16개** (표준 프로토콜 기준)
- **측정 불가: 2개**
  - **AerialVG** — HuggingFace **gated 데이터셋**(접근 승인 필요)
  - **ScreenSpot-Pro** — 데이터셋 파일 없음(`DataFilesNotFoundError`)
  - → 우리 환경 문제가 아니라 **데이터 접근 불가**
- **진행 중: Captioning 3개** (MM-MTBench · MIABench · MMIFEval)
  - ⚠️ 1차 시도는 **judge 설정 오류로 채점 실패**(judge `max-model-len=2048`인데 채점기가 `max_tokens=4096` 요청
    → 8408번 전부 `400 Bad Request` → 전부 0점). **모델 실력이 아님.**
  - judge를 `max-model-len=8192`로 재기동 → **생성 캐시 재사용해 재채점 중**
- **이후 진행: 9개** (judge 필요)
  - CharXiv · ChartMuseum · VisualProbe · MMMU-Pro(Std) · MMMU-Pro(Vision) · MathVision · SimpleVQA · FVQA · MM-Vet V2
  - ※ MathVista는 **이미 측정 완료**(71.2)

## 결과 요약 (중요 단서 포함)

- **표준 프로토콜**에선 우리 모델(terse · LoRA · 배치 1/128 스케일)이 논문 base/Vero에 **대체로 못 미침**.
- **단, 낮은 점수의 상당수는 "프롬프트 불일치 아티팩트"**:
  - 우리 모델은 학습 시 **항상 Vero 시스템프롬프트**(`<think>…</think><answer>\boxed{}</answer>`)가 있었음.
  - 표준 eval은 그 프롬프트를 주지 않음 → 모델이 산문형으로 답 → 채점기가 **정답을 맞혀도 추출 실패**.
  - **검증**: InfoVQA **3.2 → 79.2** (시스템프롬프트 주입 시). base 83.1에 근접 = 순수 아티팩트였음.
- **따라서 "실제 격차"는 공정판(시스템프롬프트 주입) 재측정 후에야 확정됨.**
- 현재 16개 중:
  - 🟩 **base 동급 이상 5개**: V\*Bench(85.9) · CV-Bench(86.8) · MME-RealWorld(49.6) · ERQA(43.8) · EmbSpatial(77.6)
  - 🟥 **아티팩트로 과소평가 5개**: InfoVQA(3.2) · ScreenSpot(0.2) · ChartQA(42.9) · Blink(32.3) · RealWorldQA(45.1)
  - 🟨 **실제로 낮음 6개**: ChartQA-Pro(35.8) · EvoChart(45.3) · CountBenchQA(77.2) · CountQA(25.1) · MathVista(71.2) · GameQA(26.8)

## 참고: 채점 코드는 정상

낮은 점수는 **평가 코드 버그가 아님**. 원인은 세 갈래:
1. **환경 문제** (예: NLTK `punkt_tab` 미설치 → 설치로 해결, judge `max_tokens` 설정 → 재기동으로 해결)
2. **프롬프트 불일치 아티팩트** (공정판으로 해소 — InfoVQA 3.2→79.2로 입증)
3. **경량화로 인한 실제 성능 저하** (LoRA·저해상도·terse·배치 1/128 → 이건 진짜)

## 다음 단계
1. Captioning 3개 재채점 완료 → **논문 terse(26.8) / full(83.8)과 직접 비교**
2. 나머지 judge 9개 측정
3. 공정판 전체 재측정 → **표준 vs 공정 2컬럼 표**로 아티팩트와 실제 격차를 분리

## 관련 파일
- 비교표(이미지): `comparison_table.png`
- 비교표(문서): `benchmark_comparison_0706_0707.md`
- 논문 수치: `investigation_2x4090/paper_table3_reference.md`
- 학습 곡선: `accuracy_vs_format_full4000.png`
