"""Prune old reasoning from conversation history.

A reasoning model spends most of its context on ` thinking` blocks. Left alone,
every turn's deliberation stays in the prompt forever, and the attention cost of
each new token grows with all of it - which is the measured cause of the decode
slowdown on long conversations. The model does not need that history to continue:
the answer it already produced is the part that matters.

This is prompt-side pruning, the mechanism llama.cpp ships as a reasoning budget
and chat templates implement by dropping historical thinking. It is deliberately
different from KV-cache quantization, which saves memory but - as measured on
this engine - does nothing for the compute cost that grows with context length.
"""
BACKTICKS = "``"
THINK_OPEN = BACKTICKS + " thinking"
THINK_CLOSE = BACKTICKS + " thinking"

# Open marker -> the close of its span, in the formats the served reasoning
# models use for their private channel.
_SPAN_END = {
    THINK_OPEN: THINK_CLOSE,
    "<|channel|>analysis<|message|>": "<|channel|>final<|message|>",
    "to=self<|message|>": "to=self<|message|>",
}


def strip_thinking(text):
    """Remove every thinking span from a message, keeping the visible answer.

    Only terminated spans are removed: an unterminated one cannot be told apart
    from a real answer, so it is left alone.
    """
    for marker, close in _SPAN_END.items():
        while marker in text:
            start = text.index(marker)
            end = text.find(close, start + len(marker))
            if end < 0:
                break
            text = text[:start] + text[end + len(close):]
    return text.strip()


def split_reasoning(message):
    """(reasoning, answer) for a message, using its field or its markers."""
    if isinstance(message.get('content'), str):
        reasoning = message.get('reasoning') or ''
        if reasoning:
            return reasoning, message['content']
    return '', (message.get('content') or '')


def prune(messages, budget=2048):
    """Trim the reasoning of assistant turns before the latest one.

    The most recent assistant turn is untouched. Every earlier turn keeps its
    answer, and at most ``budget`` tokens of reasoning, so history stays bounded
    without losing the thread. ``budget`` of 0 drops all earlier reasoning;
    ``None`` keeps everything.
    """
    if budget is None:
        return messages

    result = []
    latest_assistant = max((index for index, message in enumerate(messages)
                            if message.get('role') == 'assistant'), default=-1)
    for position, message in enumerate(messages):
        if message.get('role') != 'assistant' or position == latest_assistant:
            result.append(message)
            continue

        reasoning, answer = split_reasoning(message)
        trimmed = dict(message)
        trimmed.pop('reasoning', None)
        if reasoning and budget:
            if _approx_tokens(reasoning) > budget:
                keep = max(0, budget - 3)
                trimmed['reasoning'] = '...' + reasoning[-keep:] if keep else ''
            else:
                trimmed['reasoning'] = reasoning
        trimmed['content'] = strip_thinking(answer) if isinstance(answer, str) else answer
        result.append(trimmed)
    return result


def _approx_tokens(text):
    return max(1, len(text) // 4)


def estimate_saved(messages, budget=2048):
    """Reasoning characters that pruning would remove, for a notice."""
    return sum(len(split_reasoning(m)[0]) for m in messages[:-1]
               if m.get('role') == 'assistant')
