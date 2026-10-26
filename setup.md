# Setup: reproducing the 2 × RTX 4090 experiments

This repository is the Vero code base plus the changes needed to train and evaluate on a single workstation with two 24 GB GPUs. Everything below was run on Ubuntu 22.04, driver-level CUDA 13, 2 × RTX 4090, 128 GB host RAM.

## 1. Environment

```bash
git clone git@github.com:HyeonSung11/Difficulty-Aware-Visual-RL.git
cd Difficulty-Aware-Visual-RL
bash scripts/setup_env.sh            # creates conda env "verovlm"
conda activate verovlm
```

The script installs PyTorch 2.10 (cu130), vLLM 0.17, transformers 5.x, flash-attn, then `vero-eval` and `vero-rl` in editable mode. The exact package versions we used are in `requirements-frozen.txt`.

## 2. Locations (environment variables)

All of our scripts resolve paths through two variables so nothing is hard-wired to a machine:

| Variable | Meaning | Default |
|---|---|---|
| `VERO_REPO` | this repository | auto-detected with `git rev-parse` |
| `VERO_DATA` | data root: Vero-600k jsonl + images, checkpoints, rollout dumps, merged models | `$VERO_REPO/data` |
| `CONDA_SH` | conda init script used by the launch scripts | `$HOME/miniconda3/etc/profile.d/conda.sh` |

Expected layout under `$VERO_DATA`:

```text
$VERO_DATA/
├── vero-600k/
│   ├── vero_600k_train.verl.jsonl          # from scripts/download_and_format_vero_600k.py
│   ├── vero_600k_val.verl.jsonl
│   ├── vero_600k_train.curriculum.jsonl    # train jsonl + extra_info.difficulty_tier (step 4)
│   └── images/...
└── vero_runs/
    ├── checkpoints/verl/<experiment>/global_step_N/
    ├── rollout_dumps/<experiment>/
    └── merged_models/<experiment>/
```

## 3. Data

```bash
python scripts/download_and_format_vero_600k.py        # see docs/DATA.md
```

## 4. Difficulty labelling (k = 4)

The base model (Qwen3-VL-8B-Instruct) answers every rule-scored prompt 4 times (temperature 0.6, 262,144 max pixels, 512 new tokens) and the pass rate is binned into `easy / medium / hard / unsolved`.

```bash
bash curriculum/run_full_labeling.sh      # resumable; writes curriculum/difficulty_labels_full.jsonl
python curriculum/difficulty_crosstab.py  # tier x domain cross-table
```

Join the labels into the training jsonl as `extra_info.difficulty_tier` and save it as `vero_600k_train.curriculum.jsonl` (the launch scripts point `TRAIN_FILES` there).

Note: grounding / clicking reward types need `extra_info["image_sizes"]` for coordinate normalization; the labeller injects the loaded image size before scoring.

## 5. Training on 2 × RTX 4090

The low-resource launcher is `vero-rl/examples/model_runs/run_gspo_qwen3vl_judge_lite.sh` with config `config/gspo_qwen3vl_judge_lite.yaml`:

- GSPO, batch 4, `rollout.n = 4`, max prompt 512, max response 1,536
- LoRA rank 32 / alpha 64 on the language model only (`exclude_modules: ".*visual.*"`), actor in bf16, no offload
- vLLM rollout workers with TP = 2, `gpu_memory_utilization 0.5`, sleep level 1 between phases
- Judge: `Qwen/Qwen3-32B-AWQ` served by vLLM on the same GPUs, sleep level 2 while not scoring
- Curriculum block:

```yaml
data:
  train_files: ${oc.env:TRAIN_FILES,...vero_600k_train.curriculum.jsonl}
  sampler:
    class_path: pkg://verl/experimental/dataset/curriculum_tier_sampler
  curriculum:
    update_every: 200      # K: re-estimate live pass rate and tier weights
    easy_floor: 0.02       # minimum weight so a regressing easy tier is still watched
    min_unsolved: 0        # frontier exploration: forced unsolved prompts per batch (0 / 1 / 2)
```

```bash
export VERO_DATA=/path/to/data
export TRAIN_FILES=$VERO_DATA/vero-600k/vero_600k_train.curriculum.jsonl
export TRAINER_EXPERIMENT_NAME=curriculum_4k
cd vero-rl
bash examples/model_runs/run_gspo_qwen3vl_judge_lite.sh trainer.total_training_steps=4000
```

Variants used in the paper:

| Variant | How |
|---|---|
| terse 4k (balanced baseline) | same launcher without the `sampler` / `curriculum` block, plain `vero_600k_train.fixed.verl.jsonl` |
| curr. 4k | as above, `min_unsolved: 0`, 4,000 steps |
| curr + uns0 / uns2 | `curriculum/run_ablation_uns0.sh` / `run_catchup_uns2.sh`: resume from the curr. 4k checkpoint, `data.curriculum.min_unsolved=0|2`, `trainer.total_training_steps=6000` |
| uns1 6k (scratch) | `curriculum/chain_scratch_uns1.sh`: from the base model, `min_unsolved: 1`, 6,000 steps |

Training resumes automatically (`resume_mode=auto`) when relaunched with the same `TRAINER_EXPERIMENT_NAME`. Checkpoints are written every 200 steps.

## 6. Merge LoRA and evaluate

```bash
python investigation_2x4090/merge_lora.py \
    $VERO_DATA/vero_runs/checkpoints/verl/curriculum_4k/global_step_4000 \
    $VERO_DATA/vero_runs/merged_models/curriculum_4k

bash eval_harness/run_full_eval.sh $VERO_DATA/vero_runs/merged_models/curriculum_4k curriculum_4k
```

`run_full_eval.sh` runs the 27 benchmarks in two groups (rule-scored tasks with TP = 2, then judge-scored tasks with the GPUs split between policy and judge) under one fixed protocol: terse system prompt, `max_pixels=262144`, `max_new_tokens=512`, greedy decoding, Qwen3-32B-AWQ judge. Results land in `investigation_2x4090/eval_results_<exp_name>/`; `curriculum/collect_results.py` builds the comparison table and `curriculum/rescore_screenspot_all.py` re-scores ScreenSpot with the common 0–1000 coordinate format.

The untrained base model must be evaluated with the same protocol for a fair comparison (the `base (ours)` column in the README). Evaluating it with the direct-answer task variants instead reproduces the numbers cited in the Vero paper (e.g. CountBenchQA 89.0 vs 88.8), which is why the cited and re-measured columns differ.

## 7. Known pitfalls (from `docs/RESULTS_2x4090.md`)

- Set `actor.model_dtype: bfloat16`; otherwise the actor is loaded in fp32 and both host RAM and VRAM overflow.
- `ulysses_sequence_parallel_size: 1` (Qwen3-VL deepstack is incompatible with Ulysses SP).
- The LoRA sync path and the image-token off-by-one fix target vLLM 0.17 (`vero-rl/verl/utils/vllm/utils.py`, `verl/models/transformers/qwen3_vl.py`).
- `per_batch_domain_weights: null` because the Vero-600k `ability` labels do not match the six category weights of the original config.
