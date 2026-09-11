import unittest

from output_channels import ChannelNormaliser, normalise

MUSE = 'to=self<|message|>4 is even.<|eom|><|start|>assistant to=user<|message|>Paris'
HARMONY = '<|channel|>analysis<|message|>2+2=4.<|channel|>final<|message|>4'


class ChannelNormaliserTest(unittest.TestCase):
    def test_addressed_envelope_splits_reasoning_from_the_reply(self):
        self.assertEqual(normalise(MUSE), ' thinking4 is even.  <｜end▁of▁thinking｜>Paris')
        self.assertEqual(normalise(MUSE, split_reasoning=False), 'Paris')

    def test_harmony_envelope_splits_the_same_way(self):
        self.assertEqual(normalise(HARMONY, split_reasoning=False), '4')

    def test_text_without_markers_is_untouched(self):
        for text in ('just text', 'a < b and c > d', 'no markers here'):
            self.assertEqual(normalise(text), text)

    def test_streaming_matches_one_shot(self):
        """Marker boundaries fall inside chunks, so chunking must not change it."""
        for text in (MUSE, HARMONY):
            expected = normalise(text, split_reasoning=False)
            for size in (1, 2, 3, 5, 7, 11):
                normaliser = ChannelNormaliser(split_reasoning=False)
                pieces = [text[i:i + size] for i in range(0, len(text), size)]
                got = ''.join(normaliser.feed(piece) for piece in pieces)
                got += normaliser.feed('', final=True)
                self.assertEqual(got, expected, f'chunk size {size}')
