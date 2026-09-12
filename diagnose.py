# diagnose.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alphaedge-ai")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
)

print("\n=== 모델 배치 정보 ===")
if hasattr(model, "hf_device_map"):
    devices = set(model.hf_device_map.values())
    for k, v in model.hf_device_map.items():
        print(f"  {k}: {v}")
    print(f"\n  사용 디바이스: {devices}")
    if "cpu" in devices or "disk" in devices:
        print("  ⚠️  일부 레이어가 CPU/디스크에 있음 → 느린 원인!")
    else:
        print("  ✅ 전체 GPU에 로드됨")
else:
    print("  ✅ 전체 단일 디바이스")

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    reserv = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n  GPU 할당: {alloc:.2f} GB / 예약: {reserv:.2f} GB / 전체: {total:.1f} GB")