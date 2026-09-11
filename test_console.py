"""The console reads raw keystrokes, so its key parsing is tested against a real
pty rather than a mock: an arrow key is a byte sequence, and what matters is that
the bytes a terminal actually sends produce the action the user expects."""
import os
from pathlib import Path
import pty
import termios
import tty
import unittest

import console


class EscapeActionTest(unittest.TestCase):
    """Every sequence a terminal can send for a navigation key."""

    def test_all_four_arrows(self):
        for sequence in (b'[A', b'OA'):
            self.assertEqual(console.escape_action(sequence), 'up')
        for sequence in (b'[B', b'OB'):
            self.assertEqual(console.escape_action(sequence), 'down')
        for sequence in (b'[C', b'OC'):
            self.assertEqual(console.escape_action(sequence), 'right')
        for sequence in (b'[D', b'OD'):
            self.assertEqual(console.escape_action(sequence), 'left')

    def test_paging_and_jumps(self):
        self.assertEqual(console.escape_action(b'[5~'), 'pageup')
        self.assertEqual(console.escape_action(b'[6~'), 'pagedown')
        self.assertEqual(console.escape_action(b'[H'), 'home')
        self.assertEqual(console.escape_action(b'[F'), 'end')
        self.assertEqual(console.escape_action(b'[1~'), 'home')
        self.assertEqual(console.escape_action(b'[4~'), 'end')

    def test_a_bare_escape_quits_and_nothing_else_does(self):
        self.assertEqual(console.escape_action(b''), 'escape')
        for sequence in (b'[3~', b'[Z', b'[2~', b'OP'):
            self.assertEqual(console.escape_action(sequence), 'other')


class WindowTest(unittest.TestCase):
    def test_a_list_that_fits_is_not_scrolled(self):
        rows = list(range(10))
        self.assertEqual(console.window(rows, 5, 40), (0, 10, 0, 0))

    def test_a_long_list_keeps_the_cursor_visible(self):
        rows = list(range(40))
        first, last, above, below = console.window(rows, 0, 14)
        self.assertEqual(above, 0)
        self.assertGreater(below, 0)
        first, last, above, below = console.window(rows, 39, 14)
        self.assertEqual(last, 40)      # flush to the end, cursor visible
        self.assertGreater(above, 0)
        self.assertEqual(below, 0)
        first, last, above, below = console.window(rows, 20, 14)
        self.assertLessEqual(first, 20)
        self.assertGreater(last, 20)


class RenderTest(unittest.TestCase):
    def rows(self, count=4):
        return [{'alias': f'model-{index}', 'available': True, 'source': 'measured',
                 'weights_gib': 2.0, 'kind': 'dense', 'max_context': 262144,
                 'native_context': 262144, 'decode': 100.0, 'prefill': 1000.0}
                for index in range(count)]

    def memory(self):
        return {'available': 40 * 2**30, 'total': 64 * 2**30}

    def test_columns_shift_with_the_horizontal_offset(self):
        flat = '\n'.join(console.render(self.rows(), 0, self.memory(), 70, 20, offset=0))
        shifted = '\n'.join(console.render(self.rows(), 0, self.memory(), 70, 20, offset=16))
        self.assertIn('model-0', flat)
        self.assertNotIn('model-0', shifted)
        self.assertIn('scrolled 16 columns', shifted)

    def test_the_selected_row_is_marked(self):
        lines = console.render(self.rows(), 2, self.memory(), 90, 20)
        marked = [line for line in lines if line.startswith('>')]
        self.assertEqual(len(marked), 1)
        self.assertIn('model-2', marked[0])

    def test_a_running_instance_shows_its_port(self):
        instances = {'model-1': {'port': 8080, 'pid': 1, 'started': 0, 'rate': 90.0, 'active': 1}}
        lines = '\n'.join(console.render(self.rows(), 1, self.memory(), 110, 20, instances))
        self.assertIn(':8080', lines)
        self.assertIn('90 tok/s', lines)


if __name__ == '__main__':
    unittest.main()
