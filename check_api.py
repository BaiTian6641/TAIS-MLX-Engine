import json
from pathlib import Path
import time
import urllib.request

base = "http://127.0.0.1:8080/v1"
results = {}
for streaming in (False, True):
    payload = {
        "model": "k2-horizon",
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
        "max_tokens": 512,
        "stream": streaming,
    }
    request = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        if streaming:
            chunks = []
            done = False
            for line in response:
                if line.startswith(b"data: "):
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    chunks.append(json.loads(data))
            assert done and chunks, "Incomplete SSE response"
            content = "".join(c["choices"][0]["delta"].get("content", "") or "" for c in chunks if c.get("choices"))
            result = {"stream_complete": done, "content": content, "chunks": len(chunks)}
        else:
            result = json.load(response)
            content = result["choices"][0]["message"].get("content", "")
        assert "4" in content, f"Expected a final answer containing 4: {result}"
        results["stream" if streaming else "chat"] = {
            "seconds": time.monotonic() - start, "response": result,
        }
        print(json.dumps(results["stream" if streaming else "chat"]), flush=True)
Path(__file__).with_name("api-check.json").write_text(json.dumps(results, indent=2) + "\n")
