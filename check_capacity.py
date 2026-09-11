"""Check full-window allocation/decoding, not long-document accuracy."""
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import QuantizedKVCache

root = Path(__file__).resolve().parent
model, tokenizer = load(root / "model", trust_remote_code=True)
print("Model loaded", flush=True)
cache = []
# Synthetic history exercises every layer's full cache without hours of prefill.
for layer in range(48):
    state = QuantizedKVCache(group_size=64, bits=4)
    zeros = mx.zeros((1, 8, 262143, 128), dtype=mx.bfloat16)
    state.update_and_fetch(zeros, zeros)
    mx.eval(state.state)
    cache.append(state)
    del zeros
    mx.clear_cache()
    if (layer + 1) % 12 == 0:
        print(f"Allocated {layer + 1}/48 full-length caches", flush=True)
start = time.monotonic()
logits = model(mx.array([[tokenizer.encode("Hello")[0]]]), cache=cache)
mx.eval(logits)
assert bool(mx.all(mx.isfinite(logits)).item()), "Non-finite logits"
assert all(c.offset == 262144 for c in cache)
result = {
    "test": "synthetic full-window cache allocation and one forward pass",
    "context_tokens": 262144,
    "kv_bits": 4,
    "cache_gib": sum(c.nbytes for c in cache) / 2**30,
    "peak_mlx_gib": mx.get_peak_memory() / 2**30,
    "forward_seconds": time.monotonic() - start,
    "finite_logits": True,
    "full_document_prefill_tested": False,
}
(root / "capacity-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
