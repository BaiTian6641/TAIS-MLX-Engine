"""Vision input handling and image-token expansion.

The fast tests (decoding, part normalisation) run anywhere. The token-expansion
test loads only the checkpoint's tokenizer and image processor - not the 14 GB
of weights - so it stays quick, and it skips when the checkpoint is absent.
End-to-end image *generation* is covered by ``check_vision.py``.
"""

import base64
import io
import os
import unittest
from pathlib import Path

from input_parts import (
    ContentPartError,
    decode_image,
    extract_vision_messages,
    normalize_message_content,
)

GEMMA = Path(__file__).parent / "models" / "gemma4-26b-a4b"


def _png_b64(color=(200, 0, 0), size=(32, 32)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class DecodeImageTest(unittest.TestCase):
    def test_data_url_and_raw_base64(self):
        b64 = _png_b64()
        for source in (f"data:image/png;base64,{b64}", b64):
            img = decode_image(source)
            self.assertEqual(img.mode, "RGB")
            self.assertEqual(img.size, (32, 32))
            self.assertEqual(img.getpixel((0, 0)), (200, 0, 0))

    def test_bad_payload_raises(self):
        with self.assertRaises(ContentPartError):
            decode_image("not-a-real-image")


class ExtractVisionMessagesTest(unittest.TestCase):
    def test_image_part_becomes_template_marker_and_is_decoded(self):
        b64 = _png_b64()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "tool_use", "name": "exec", "arguments": {"cmd": "ls"}},
            ]}
        ]
        out, images = extract_vision_messages(messages)
        self.assertEqual(len(images), 1)
        kinds = [p["type"] for p in out[0]["content"]]
        self.assertEqual(kinds, ["text", "image", "text"])
        # The tool call is flattened to text; the image stays a marker.
        self.assertIn("Tool call: exec", out[0]["content"][2]["text"])

    def test_unsupported_part_names_its_type(self):
        with self.assertRaises(ContentPartError) as ctx:
            extract_vision_messages([{"role": "user", "content": [{"type": "video_file", "f": 1}]}])
        self.assertIn("video_file", str(ctx.exception))


class NormalizeMessageContentTest(unittest.TestCase):
    def test_text_path_flattens_image_to_placeholder(self):
        b64 = _png_b64()
        out, images = normalize_message_content([
            {"role": "user", "content": [
                {"type": "text", "text": "Look: "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}
        ])
        self.assertEqual(len(images), 1)
        self.assertEqual(out[0]["content"], "Look: <|image_pad|>")


@unittest.skipUnless(GEMMA.is_dir(), "gemma4-26b-a4b checkpoint not present")
class ImageTokenExpansionTest(unittest.TestCase):
    """build_inputs expands <|image|> to boi + soft-tokens + eoi, model-free."""

    @classmethod
    def setUpClass(cls):
        from vision_engine import VisionModel, load_vision_model  # noqa: F401
        from mlx_lm.utils import load_tokenizer
        from vendor.flash_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor

        cls.vm = VisionModel(
            model=None,
            tokenizer=load_tokenizer(str(GEMMA), tokenizer_config_extra={"trust_remote_code": True}),
            image_processor=Gemma4ImageProcessor(),
            config=None,
        )

    def test_expansion_places_soft_tokens_between_boi_and_eoi(self):
        tok = self.vm.tokenizer
        b64 = _png_b64(color=(255, 220, 0), size=(256, 256))
        messages, images = extract_vision_messages([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Describe it."},
            ]}
        ])
        self.assertEqual(len(images), 1)
        input_ids, pixel_values = self.vm.build_inputs(messages, images)
        ids = input_ids.tolist()

        boi, soft, eoi = tok.boi_token, tok.image_token_id, tok.eoi_token
        boi_id = tok.convert_tokens_to_ids(boi) if isinstance(boi, str) else boi
        eoi_id = tok.convert_tokens_to_ids(eoi) if isinstance(eoi, str) else eoi
        self.assertIn(boi_id, ids)
        self.assertIn(eoi_id, ids)
        n_soft = ids.count(soft)
        self.assertGreater(n_soft, 0)
        # Soft tokens form one contiguous run wrapped by boi/eoi.
        run = ids[ids.index(boi_id): ids.index(eoi_id) + 1]
        self.assertEqual(run[0], boi_id)
        self.assertEqual(run[-1], eoi_id)
        self.assertTrue(all(t == soft for t in run[1:-1]))
        self.assertIsNotNone(pixel_values)


if __name__ == "__main__":
    unittest.main()
