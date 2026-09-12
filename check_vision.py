"""End-to-end vision check: load a VLM profile and verify it sees.

Runs a small battery of verifiable images (colour, OCR, counting) through the
full ``vision_engine`` path and writes ``<profile>-vision-check.json`` with the
answers and timings. This is the measurement companion to ``test_vision.py``
(which covers the model-free logic); it loads real weights.

    .venv/bin/python check_vision.py [--model gemma4-26b-a4b]
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from vision_engine import VISION_PROFILES, load_vision_model


def _images():
    # Solid-colour circle on a contrasting background.
    circle = Image.new("RGB", (512, 512), (255, 220, 0))
    ImageDraw.Draw(circle).ellipse([96, 96, 416, 416], fill=(200, 0, 0))
    # A single large digit.
    digit = Image.new("RGB", (400, 400), (255, 255, 255))
    ImageDraw.Draw(digit).text((140, 120), "7", fill=(0, 0, 0))
    # Three squares.
    squares = Image.new("RGB", (512, 256), (255, 255, 255))
    d = ImageDraw.Draw(squares)
    for x in (40, 150, 260):
        d.rectangle([x, 80, x + 70, 150], fill=(0, 0, 200))
    return circle, digit, squares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-26b-a4b", choices=sorted(VISION_PROFILES))
    args = ap.parse_args()
    model_dir = Path("models") / args.model

    t0 = time.perf_counter()
    vm = load_vision_model(str(model_dir))
    load_s = time.perf_counter() - t0

    circle, digit, squares = _images()
    cases = [
        ("colour", circle,
         "What color is the circle and the background? One short sentence.", ("red", "yellow")),
        ("ocr", digit, "What number is shown? Just the digit.", ("7",)),
        ("count", squares, "How many blue squares? Just the number.", ("3",)),
    ]

    results = []
    for name, image, prompt, expect in cases:
        t1 = time.perf_counter()
        out = vm.generate(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            images=[image], max_tokens=160)
        dt = time.perf_counter() - t1
        ok = all(e.lower() in out.lower() for e in expect)
        results.append({"case": name, "answer": out, "expected": expect,
                        "ok": ok, "seconds": round(dt, 2)})
        print(f"[{name}] {'OK ' if ok else 'MISS'} ({dt:.1f}s) -> {out!r}")

    passed = sum(r["ok"] for r in results)
    report = {"model": args.model, "load_seconds": round(load_s, 1),
              "passed": passed, "total": len(results), "results": results}
    out_path = Path(f"{args.model}-vision-check.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n{passed}/{len(results)} passed; wrote {out_path}")


if __name__ == "__main__":
    main()
