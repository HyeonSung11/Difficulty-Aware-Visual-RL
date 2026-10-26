# <think> prefill 진단 결과 (2026-07-01)

프롬프트 assistant를 `<think>`로 prefill 후 base Qwen3-VL-8B 생성 (temp 0.6, 4샘플):

| 샘플 | 출력 요약 | think쌍 | answer쌍 | boxed | 정답? |
|---|---|---|---|---|---|
| 0 | `<think> ...grebe... </answer><answer>[point_2d]` | ✗ | ✗ | ✗ | - |
| 1 | `<think> ...Groundhog... </answer>\boxed{Groundhog Day}` | ✗ | ✗ | ✓ | ✓(내용) |
| 2 | `<think> ...value 2000. </answer><answer> \boxed{2000} </answer>` | ✗ | ✓ | ✓ | ✓ (gt 2000) |
| 3 | `<think> ...three colors. </answer><answer> \boxed{A} </answer>` | ✗ | ~ | ✓ | ✓ (gt A) |

**핵심 발견**: `<think>` prefill해도 모델이 **`</think>` 대신 `</answer>`로 thinking을 닫음**.
→ `<think></think>` 완전쌍 0/4 → 현재 format_reward(=<think></think> 필수)는 여전히 0점.
그러나 **끝의 `<answer> \boxed{X} </answer>`와 정답 내용은 정확히 생성**함.

**결론**: prefill 단독 불충분. 모델은 Instruct(non-thinking)라 think/answer 태그를 혼동.
→ format_reward를 완화(`</think>` 강제 해제, `<answer>\boxed{}` 기준)해야 학습 신호 발생.
