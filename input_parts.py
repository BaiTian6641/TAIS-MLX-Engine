"""Uniform content-part handling for the serving layer.

Agentic clients (oh-my-pi, Claude-style tools, OpenAI multipart) send message
content as a *list of parts*, not a string. The pinned ``mlx_lm`` server
accepts only ``{"type": "text"}`` parts and raises on everything else, which
breaks every real agentic conversation (``tool_use``/``tool_result``) and any
multimodal request.

This module normalises message content into a canonical form:
- ``text``            -> its own text
- ``tool_use``        -> ``Tool call: name(arguments)``
- ``tool_result``     -> the nested content
- ``input_text`` / a ``text`` field on any part -> that text
- ``image_url`` / ``image`` / ``input_image`` -> decoded to a PIL image

Everything a *text-only* model cannot consume is reported with the specific
part type, not a generic "Only 'text' content type is supported".
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is a serving dependency
    Image = None

__all__ = ["normalize_message_content", "decode_image", "ContentPartError"]


class ContentPartError(ValueError):
    """A content part the model cannot consume."""


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def decode_image(source: Any) -> "Image.Image":
    """Decode an image source into a PIL image.

    Accepts a base64 ``data:`` URL, an ``http(s)`` URL, a raw base64 string, or
    a PIL image that is already decoded.
    """
    if Image is None:  # pragma: no cover
        raise ContentPartError("Pillow is required to decode image content")
    if isinstance(source, dict):
        source = source.get("url") or source.get("data") or source.get("image") or source.get("base64")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if not isinstance(source, str):
        raise ContentPartError(f"Unsupported image source: {type(source).__name__}")
    if source.startswith("data:"):
        source = source.split(",", 1)[1] if "," in source else source
        return _decode_b64(source)
    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(source, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return _decode_b64(source)


def _decode_b64(payload: str) -> "Image.Image":
    try:
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    except Exception as exc:  # pragma: no cover - malformed input
        raise ContentPartError(f"Could not decode base64 image: {exc}") from exc


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------

def _part_text(part: Dict[str, Any], images: List["Image.Image"]) -> str:
    kind = part.get("type")
    if kind in (None, "text", "input_text"):
        return str(part.get("text", ""))
    if kind == "tool_use":
        return f"Tool call: {part.get('name')}({part.get('arguments') or part.get('input') or ''})"
    if kind == "tool_result":
        return _stringify_nested(part.get("content"))
    if kind in ("image_url", "image", "input_image"):
        images.append(decode_image(part.get("image_url") or part.get("image") or part.get("url") or part))
        return "<|image_pad|>"
    if "text" in part:
        return str(part["text"])
    raise ContentPartError(
        f"Content part of type '{kind}' is not supported by this model."
    )


def _stringify_nested(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_stringify_nested(c) for c in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    return str(content) if content is not None else ""


def normalize_message_content(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List["Image.Image"]]:
    """Return ``(messages, images)`` with every message's content a string.

    ``messages`` is a copy with ``content`` flattened to text; ``images`` holds
    the decoded images in encounter order (for a vision-capable model). A
    part no model can consume raises :class:`ContentPartError` naming its type.
    """
    images: List["Image.Image"] = []
    out: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            out.append({**message, "content": "".join(_part_text(p, images) for p in content)})
        elif content is None:
            out.append({**message, "content": ""})
        else:
            out.append(message)
    return out, images


def normalize_tool_call_arguments(message: Dict[str, Any]) -> Dict[str, Any]:
    """Make a message's ``tool_calls`` arguments mappings, in place.

    OpenAI's wire format carries ``function.arguments`` as a JSON *string*, but
    chat templates iterate it as a mapping (``tool_call.arguments|items`` in the
    Qwen template), which raises "Can only get item pairs from a mapping" on a
    string. A missing key is fine (the filter skips ``Undefined``); ``null`` has
    to become ``{}``.
    """
    for tool_call in message.get("tool_calls") or []:
        func = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not func:
            continue
        args = func.get("arguments")
        if isinstance(args, str):
            try:
                func["arguments"] = json.loads(args)
            except json.JSONDecodeError:
                func["arguments"] = {"value": args}
        elif args is None and "arguments" in func:
            func["arguments"] = {}
    return message


def extract_vision_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List["Image.Image"]]:
    """Return ``(messages, images)`` ready for a vision chat template.

    Unlike :func:`normalize_message_content` (which flattens images to a text
    placeholder for a text-only model), this keeps each image as the
    ``{"type": "image"}`` part the template expands into soft tokens, and
    decodes the images in encounter order. Text, tool and other consumable
    parts are flattened to ``{"type": "text"}`` parts.
    """
    images: List["Image.Image"] = []
    out: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for part in content:
                kind = part.get("type")
                if kind in ("image_url", "image", "input_image"):
                    images.append(decode_image(part.get("image_url") or part.get("image") or part.get("url") or part))
                    parts.append({"type": "image"})
                else:
                    parts.append({"type": "text", "text": _part_text(part, images)})
            message = {**message, "content": parts}
        elif content is None:
            message = {**message, "content": ""}
        out.append(normalize_tool_call_arguments(message))
    return out, images
