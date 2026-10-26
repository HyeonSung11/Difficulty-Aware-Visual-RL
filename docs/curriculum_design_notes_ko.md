# Vero 커리큘럼 러닝 — 최종 학습 과정 정리

> base 모델 pass@4 난이도 라벨(600K)을 기반으로, 학습 중 tier 가중치를 `p(1−p)`로
> 동적 재분배하는 온라인 커리큘럼. 목표: 같은 스텝 예산에서 낭비를 줄여 Vero 초월.

---

## 0. 준비물 (완료 ✅)
- **난이도 라벨 600,000개**: `curriculum/difficulty_labels_full.jsonl` — base 모델 pass@4로 각 프롬프트에 tier 부여
- **tier 정의**: easy(4/4) · medium(2~3/4) · hard(1/4) · unsolved(0/4)
- **분포**: easy 18.9% / medium 21.7% / hard 9.9% / unsolved 49.5%
- 도메인별: chart_ocr 0.46 > knowledge 0.40 > stem 0.29 > counting_grounding 0.22 > spatial 0.15 (평균 pass rate)

## 1. 셋업
| 항목 | 값 |
|---|---|
| 베이스 | Qwen3-VL-8B-Instruct |
| 방식 | LoRA(r32, α64, LM-only) + GSPO, terse 리워드 |
| 하드웨어 | 2×4090 |
| 스텝 예산 | 4000 (하드캡) |

## 2. 데이터에 tier 태그 부착
- `difficulty_labels_full.jsonl`의 tier를 학습 데이터(`vero_600k_train`)의 `extra_info.tier`에 id 조인 merge

## 3. 커리큘럼 핵심 규칙 — `p(1−p)` 연속 가중

매 K step(예 200)마다 **4개 tier를 전부 측정하고 전부 재분배**:

```
각 tier t:
    p[t] = 최근 K step 롤아웃의 tier t 라이브 정답률
    signal[t] = p[t] * (1 - p[t])     # 0.5에서 최대, 0/1에서 0
    w[t] = signal[t] / sum(signal)    # 정규화 (합=1)
```

| 라이브 정답률 | p(1−p) | 무게 |
|---|---|---|
| 0.5 (반반) | 0.25 (최대) | 높음 |
| 0.8 | 0.16 | 중간 |
| 0.95 / 0.05 | 낮음 | 낮음 |
| 1.0 / 0.0 (다맞/다틀) | 0 | 0 |

→ 항상 "지금 반반 맞히는 tier"에 자동으로 무게가 실림.

## 4. 학습 흐름 (자동·양방향)

```
매 200 step: 4개 tier 정답률 전부 측정 → w 전부 재계산

초반:  medium p≈0.5 → 무게 최대 (폭발적 학습)
  ↓ medium 잘하게 됨 (p 0.5→0.85) → medium 무게↓, hard 무게↑
  ↓ hard 학습 → 실력↑ → 예전 unsolved가 p 0→0.3 → unsolved 무게 자동↑
  ↓ medium 까먹으면 (p 0.85→0.6) → medium 무게 자동 부활 ★

★ 단방향 아님. 어느 tier든 퇴행하면(p가 0.5쪽으로) 즉시 무게 회복.
* 각 tier 내 6도메인 균등 유지 (Vero 핵심: 다양성이 결정적)
```

## 5. 종료
| 조건 | 설명 |
|---|---|
| 하드캡(주) | 4000 step 도달 — 커리큘럼은 학습 길이를 늘리지 않음 |
| 자연 수렴 | 모든 tier p→1 or 0 → 모든 무게→0 → 배울 것 없음 |
| 조기종료(옵션) | reward·전 tier 정답률 정체 시 |

## 6. 평가 (기존 harness 재사용)
- 학습 LoRA 병합 → `eval_harness/run_full_eval.sh <경로>` 한 줄 → 27벤치 자동 측정
- 베이스라인 표(Ours-v1)에 **Ours-v2 열 추가**해 개선폭 정량화

## 7. 구현 부품 (verl/Vero 추가)
1. **tier 태그 merge** — 데이터 전처리 (id 조인)
2. **가중 샘플러** — `w[tier]` 비율로 프롬프트 추출 (rl_dataset)
3. **`p(1−p)` 콜백** — 매 K step tier별 정답률 집계 → 무게 재계산 (trainer hook)
4. (+) **`filter_groups`(방안 2)** 병행 — step 내 잔여 zero-advantage 그룹 미세 제거

---

## 핵심 로직 (한 문장씩)
1. 다 맞는 것(easy)·다 틀리는 것(unsolved)은 학습신호 0 → 무게 자동 0
2. 매 주기 4개 tier 전부 측정해 `p(1−p)`로 무게 재분배 (한 번 졸업해도 계속 감시)
3. 반반 맞히는 tier에 무게 집중 → 최대 학습
4. 실력 늘면 unsolved도 반반으로 올라와 자동 편입, 퇴행하면 자동 부활
5. 같은 4000 스텝을 낭비 없이 → Vero 대비 성능↑ (또는 더 빨리 도달)

## Vero 대비 차별점 (= 기여)
| | Vero | 우리 커리큘럼 |
|---|---|---|
| 데이터 편성 | 무작위 (난이도 무시) | `p(1−p)` 동적 가중 |
| zero-advantage | 매 스텝 ~50% 낭비 | 자동 0 무게 + 동적필터 |
| 퇴행 대응 | 없음 | 양방향 자가교정 |
| 저자원 효율 | — | batch4×n4에서 특히 큰 이득 |

---

## 배경 개념
- **learnable zone**: base가 2~3/4·1/4 맞히는 구간(medium+hard). RL은 여기서만 배움.
- **왜 easy/unsolved는 안 배우나**: GSPO advantage = 그룹 내 리워드 차이. 4개가 다 정답/다 오답이면 차이 0 → gradient 0.
- **난이도는 정책 상대값**: base 라벨은 시작점일 뿐, 학습하면서 라이브 정답률로 갱신.
