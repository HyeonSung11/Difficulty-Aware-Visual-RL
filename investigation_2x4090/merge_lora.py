"""
최종 LoRA 어댑터(global_step_4000)를 base Qwen3-VL-8B에 병합해 eval용 merged 모델 저장.
usage: python merge_lora.py <adapter_dir> <out_dir>
"""
import sys, torch
from peft import PeftModel
from transformers import AutoProcessor

adapter_dir = sys.argv[1]
out_dir = sys.argv[2]
BASE = "Qwen/Qwen3-VL-8B-Instruct"

# 모델 클래스 로드 (Qwen3-VL)
try:
    from transformers import AutoModelForImageTextToText as AutoM
except Exception:
    from transformers import AutoModelForCausalLM as AutoM

print("base 로드...", flush=True)
base = AutoM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
print("어댑터 적용...", flush=True)
model = PeftModel.from_pretrained(base, adapter_dir)
print("merge_and_unload...", flush=True)
merged = model.merge_and_unload()
print("저장...", flush=True)
merged.save_pretrained(out_dir, safe_serialization=True)
AutoProcessor.from_pretrained(BASE).save_pretrained(out_dir)
print(f"DONE merged -> {out_dir}", flush=True)
