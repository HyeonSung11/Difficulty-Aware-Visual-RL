# Vero 커리큘럼 러닝 — 난이도 tier 동적 가중 샘플러
#
# 원리: 매 K step 마다 tier별 라이브 정답률 p 를 측정하고, 각 tier 무게를
#       p(1-p) (0.5에서 최대, 0/1에서 0) 로 재분배한다. 항상 "지금 반반 맞히는"
#       tier에 무게가 실리고, 어느 tier가 퇴행하면(p가 0.5쪽으로) 자동 부활.
#
# 데이터: extra_info.difficulty_tier ∈ {easy, medium, hard, unsolved}
# 정답률: update(batch) 에서 batch.non_tensor_batch['acc'] (샘플별) 로 집계.
#
# 사용(config):
#   data:
#     sampler:
#       class_path: verl.experimental.dataset.curriculum_tier_sampler
#       class_name: CurriculumTierSampler
#     dataloader_num_workers: 0     # 커리큘럼 샘플러 필수
#     curriculum:
#       update_every: 200
#       init_weights: {easy: 0.05, medium: 0.75, hard: 0.20, unsolved: 0.0}
#       easy_floor: 0.02            # easy 퇴행 감시용 최소 무게
import sys

import numpy as np

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

TIERS = ["easy", "medium", "hard", "unsolved"]


class CurriculumTierSampler(AbstractCurriculumSampler):
    def __init__(self, data_source, data_config):
        self.dataset = data_source
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
        self.seed = int(data_config.get("seed", 18))
        self.shuffle = bool(data_config.get("shuffle", True))

        cur = data_config.get("curriculum", {}) or {}
        self.K = int(cur.get("update_every", 200))
        self.easy_floor = float(cur.get("easy_floor", 0.02))
        # unsolved(base가 0/4로 전부 틀린 프론티어)를 배치마다 최소 N개 강제 편입.
        # batch=4에서 floor 무게(0.04)는 0개로 반올림돼 한 번도 안 뽑히므로,
        # 프론티어 탐색을 원하면 여기서 하한을 준다(나머지는 medium/hard에서 차감).
        self.min_unsolved = int(cur.get("min_unsolved", 0))
        init = dict(cur.get("init_weights", {}) or {})
        self.weights = {t: float(init.get(t, {"easy": 0.05, "medium": 0.75,
                                              "hard": 0.20, "unsolved": 0.0}[t])) for t in TIERS}

        # ---- tier -> indices (dataframe에서 싼값으로 읽음) ----
        tiers_col = self._read_tiers()
        self.tier2indices = {t: [] for t in TIERS}
        for i, t in enumerate(tiers_col):
            self.tier2indices[t if t in self.tier2indices else "medium"].append(i)
        self.tier2indices = {t: np.array(v, dtype=np.int64) for t, v in self.tier2indices.items()}
        print("[curriculum] tier 분포:",
              {t: len(v) for t, v in self.tier2indices.items()}, file=sys.stderr, flush=True)

        self.rng = np.random.default_rng(self.seed)
        self._perm = {}
        self._pos = {}
        for t in TIERS:
            self._reshuffle(t)
        self._num_samples = (len(self.dataset) // self.batch_size) * self.batch_size
        self._recompute_batch_counts()

        # 라이브 정답률 누적기
        self.acc_sum = {t: 0.0 for t in TIERS}
        self.acc_cnt = {t: 0 for t in TIERS}
        self._steps = 0

    # ------- tier 읽기 -------
    def _read_tiers(self):
        df = getattr(self.dataset, "dataframe", None)
        if df is not None and "extra_info" in getattr(df, "column_names", []):
            return [(ei or {}).get("difficulty_tier", "medium") for ei in df["extra_info"]]
        # 폴백: __getitem__ 없이 못 읽으면 전부 medium
        return ["medium"] * len(self.dataset)

    # ------- 무게 -> 배치당 tier별 개수 -------
    def _recompute_batch_counts(self):
        w = np.array([max(self.weights[t], 0.0) for t in TIERS], dtype=np.float64)
        # 인덱스가 없는 tier는 무게 0
        for i, t in enumerate(TIERS):
            if len(self.tier2indices[t]) == 0:
                w[i] = 0.0
        if w.sum() <= 0:
            w = np.array([len(self.tier2indices[t]) for t in TIERS], dtype=np.float64)
        w = w / w.sum()
        raw = w * self.batch_size
        counts = np.floor(raw).astype(np.int64)
        leftover = int(self.batch_size - counts.sum())
        if leftover > 0:
            order = np.argsort(-(raw - counts))
            for j in range(leftover):
                counts[order[j % len(order)]] += 1

        # ---- unsolved 최소 보장: 하한만큼 medium/hard에서 빼서 채운다 ----
        u = TIERS.index("unsolved")
        need = min(self.min_unsolved, self.batch_size)
        if need > 0 and len(self.tier2indices["unsolved"]) > 0 and counts[u] < need:
            donors = [TIERS.index("medium"), TIERS.index("hard"),
                      TIERS.index("easy")]  # 이 순서로 큰 곳부터 차출
            while counts[u] < need:
                # 차출 가능한(개수>0) tier 중 가장 많은 곳에서 1개 이동
                cand = [i for i in donors if counts[i] > 0]
                if not cand:
                    break
                src = max(cand, key=lambda i: counts[i])
                counts[src] -= 1
                counts[u] += 1
        self._batch_counts = {TIERS[i]: int(counts[i]) for i in range(len(TIERS))}

    def _reshuffle(self, t):
        idx = self.tier2indices[t]
        self._perm[t] = self.rng.permutation(idx) if self.shuffle else idx.copy()
        self._pos[t] = 0

    def _take(self, t, count):
        out = []
        if len(self.tier2indices[t]) == 0:
            return out
        while len(out) < count:
            perm, pos = self._perm[t], self._pos[t]
            take = min(count - len(out), len(perm) - pos)
            if take > 0:
                out.extend(perm[pos:pos + take]); pos += take
            if pos >= len(perm):
                self._reshuffle(t); pos = self._pos[t]
            self._pos[t] = pos
        return out

    def __iter__(self):
        num_batches = self._num_samples // self.batch_size
        for _ in range(num_batches):
            batch = []
            for t in TIERS:
                c = self._batch_counts.get(t, 0)
                if c:
                    batch.extend(self._take(t, c))
            if self.shuffle:
                self.rng.shuffle(batch)
            for idx in batch:
                yield int(idx)

    def __len__(self):
        return self._num_samples

    # ------- 매 스텝 피드백: 정답률 집계 + K마다 무게 갱신 -------
    def update(self, batch: DataProto) -> None:
        ntb = batch.non_tensor_batch
        # 첫 호출 1회: 실제 키를 찍어서 acc 소스 확인 (디버그)
        if not getattr(self, "_dbg_done", False):
            try:
                bk = list(getattr(batch, "batch", {}).keys()) if hasattr(batch, "batch") else []
                print(f"[curriculum][dbg] non_tensor keys={list(ntb.keys())} | tensor keys={bk}",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[curriculum][dbg] key dump err: {e}", file=sys.stderr, flush=True)
            self._dbg_done = True

        # 정답률(acc) 소스: non_tensor의 acc/accuracy -> 없으면 reward tensor로 대체
        accs = ntb.get("acc")
        if accs is None:
            accs = ntb.get("accuracy")
        if accs is None:
            # fallback: 시퀀스 리워드 부호로 근사 (>0 이면 정답으로 간주)
            for key in ("token_level_scores", "token_level_rewards", "rm_scores"):
                tb = getattr(batch, "batch", {})
                if key in tb:
                    try:
                        accs = (tb[key].sum(-1) > 0).float().cpu().numpy()
                    except Exception:
                        accs = None
                    break

        exs = ntb.get("extra_info")
        tiers = None
        if exs is not None:
            tiers = [(ei or {}).get("difficulty_tier", "medium") for ei in exs]
        elif "difficulty_tier" in ntb:
            tiers = list(ntb["difficulty_tier"])

        if accs is not None and tiers is not None and len(accs) == len(tiers):
            for a, t in zip(accs, tiers):
                if t not in self.acc_sum:
                    t = "medium"
                self.acc_sum[t] += float(a)
                self.acc_cnt[t] += 1

        self._steps += 1
        if self._steps >= self.K:
            self._recompute_weights()
            self._steps = 0

    def _recompute_weights(self):
        sig = {}
        live = {}
        for t in TIERS:
            p = (self.acc_sum[t] / self.acc_cnt[t]) if self.acc_cnt[t] > 0 else None
            live[t] = p
            if p is None:
                sig[t] = self.easy_floor              # 측정 없으면 floor만 (탐색)
            else:
                # 모든 tier에 floor 적용 (unsolved 편입·easy 감시 = 탐색 보장)
                sig[t] = max(p * (1.0 - p), self.easy_floor)
        S = sum(sig.values()) or 1.0
        self.weights = {t: sig[t] / S for t in TIERS}
        self._recompute_batch_counts()
        print(f"[curriculum] weights 갱신 live_p={ {t: (round(live[t],3) if live[t] is not None else None) for t in TIERS} } "
              f"-> w={ {t: round(self.weights[t],3) for t in TIERS} }", file=sys.stderr, flush=True)
        self.acc_sum = {t: 0.0 for t in TIERS}
        self.acc_cnt = {t: 0 for t in TIERS}
