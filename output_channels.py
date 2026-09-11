"""Normalise models that address their output into channels.

Two of the served models do not emit plain text. GPT-OSS wraps its answer in a
harmony envelope (``<|channel|>analysis<|message|>…<|channel|>final<|message|>…``)
and Muse Glimmer addresses every message (``<|start|>assistant to=self<|message|>``
for its own reasoning, ``to=user<|message|>`` for the reply). Both are answering
correctly; a server that forwards the raw stream shows the user template markers
and the model's private reasoning instead of the answer.

This rewrites those envelopes into the `` thinking`` / ``<｜end▁of▁thinking｜>`` convention the
runtime's text state machine already understands, so the reply lands in the
content field and the reasoning in the reasoning field. When the tokenizer does
not declare thinking support there is nothing to split into, so the internal
channel is dropped instead of being shown.

Markers straddle token boundaries, so the transformer is stateful and buffers a
possible partial marker at the end of each chunk.
"""
import re

# Markers that open or close a channel, in the order they must be recognised.
_INTERNAL_OPEN = ('<|channel|>analysis<|message|>', 'to=self<|message|>',
                  '<|channel|>analysis', 'to=self')
_PUBLIC_OPEN = ('<|channel|>final<|message|>', 'to=user<|message|>',
                '<|channel|>final', 'to=user')
_TRAILING = ('<|start|>assistant', '<|end|>', '<|eom|>', '<|eot|>', '<|return|>',
             '<|channel|>', '<|message|>', '<|begin_of_text|>')

MARKERS = _INTERNAL_OPEN + _PUBLIC_OPEN + _TRAILING
_ORDERED = ([(marker, 'internal') for marker in _INTERNAL_OPEN]
            + [(marker, 'public') for marker in _PUBLIC_OPEN]
            + [(marker, None) for marker in _TRAILING])
_LOOKAHEAD = max(len(marker) for marker in MARKERS)


class ChannelNormaliser:
    """Rewrite channel envelopes in a stream of text chunks."""

    def __init__(self, split_reasoning=True):
        self.split_reasoning = split_reasoning
        self.channel = None          # None, 'internal' or 'public'
        self.seen_output = False
        self.pending = ''

    def _find(self):
        """Earliest marker in the buffer, preferring the longest at that index."""
        best = None
        for marker, channel in _ORDERED:
            index = self.pending.find(marker)
            if index < 0:
                continue
            if best is None or index < best[0] or (index == best[0] and len(marker) > len(best[1])):
                best = (index, marker, channel)
        return best

    def _completing(self, index, marker):
        rest = self.pending[index:]
        return any(len(other) > len(marker) and other.startswith(rest) for other in MARKERS)

    def _holdback(self):
        """Length of a tail that could still grow into a marker, if any."""
        tail = min(len(self.pending), _LOOKAHEAD)
        for length in range(tail, 0, -1):
            suffix = self.pending[-length:]
            if any(marker.startswith(suffix) for marker in MARKERS):
                return length
        return 0

    def feed(self, chunk, final=False):
        """Return the text to emit for this chunk, buffering partial markers."""
        self.pending += chunk
        out = []
        while self.pending:
            found = self._find()
            if found is None:
                keep = 0 if final else self._holdback()
                piece = self.pending[:len(self.pending) - keep] if keep else self.pending
                self.pending = self.pending[len(piece):]
                out.append(self._emit(piece))
                break
            index, marker, channel = found
            if not final and self._completing(index, marker):
                # The buffer ends inside a longer marker that this prefix could
                # still grow into - wait for it rather than strip the short one.
                break
            out.append(self._emit(self.pending[:index]))
            self.pending = self.pending[index + len(marker):]
            if channel == 'internal':
                self.channel = 'internal'
                if self.split_reasoning:
                    out.append(' thinking')
            elif channel == 'public':
                self.channel = 'public'
                if self.split_reasoning:
                    out.append(' <｜end▁of▁thinking｜>')
        if final and self.pending:
            out.append(self._emit(self.pending))
            self.pending = ''
        return ''.join(out)

    def _emit(self, text):
        if not text:
            return ''
        if self.channel == 'internal' and not self.split_reasoning:
            return ''
        return text

def normalise(text, split_reasoning=True):
    """One-shot form, for tests and non-streaming callers."""
    return ChannelNormaliser(split_reasoning).feed(text, final=True)


def addressed(tokenizer):
    """Only models whose templates address channels need the rewrite.

    Gating on ``has_thinking`` as well would mangle models that never emit an
    envelope but do quote one: a literal ``to=self<|message|>`` in the text of a
    thinking model would be rewritten into a thinking marker.
    """


def _addressed(tokenizer):
    return bool(re.search(r'<\|channel\|>|to=user<\|message\|>|<\|start\|>assistant',
                          getattr(tokenizer, 'chat_template', None) or ''))
