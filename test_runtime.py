import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from runtime_support import ContextPolicy, GIB
from telemetry import Telemetry


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ContextPolicy(json.loads(Path('model/config.json').read_text()))

    def test_memory_and_native_bounds(self):
        self.assertEqual(self.policy.bytes_per_token, 55296)
        self.assertEqual(self.policy.capacity(100*GIB, 0, 100*GIB), 524288)
        large = self.policy.capacity(40*GIB, 20*GIB, 52*GIB)
        small = self.policy.capacity(20*GIB, 20*GIB, 52*GIB)
        self.assertGreater(large, 262144)
        self.assertLess(small, large)
        self.assertEqual(self.policy.capacity(GIB, 20*GIB, 52*GIB), 0)

    def test_output_clipping_and_prompt_rejection(self):
        self.assertEqual(self.policy.admit(300000, 32768, 310000), 10000)
        self.assertEqual(self.policy.admit(309999, 100, 310000), 1)
        with self.assertRaises(ValueError):
            self.policy.admit(310000, 1, 310000)

    def test_queue_lifecycle(self):
        telemetry = Telemetry()
        a, b = SimpleNamespace(), SimpleNamespace()
        ka, kb = telemetry.submit(a), telemetry.submit(b)
        telemetry.update(ka, phase='prefill')
        telemetry.token(ka)
        snap = telemetry.snapshot()
        self.assertEqual((snap['running'], snap['queued']), (1, 1))
        telemetry.finish(ka)
        telemetry.finish(kb, error=True)
        self.assertEqual(telemetry.snapshot()['running'], 0)
        self.assertEqual((telemetry.completed, telemetry.failed), (1, 1))


class DiskTests(unittest.TestCase):
    def test_roundtrip_branch_restart_and_eviction(self):
        import mlx.core as mx
        from mlx_lm.models.cache import QuantizedKVCache
        from disk_cache import DiskPromptCache
        with tempfile.TemporaryDirectory() as directory:
            disk = DiskPromptCache(directory, 'test', max_bytes=10**7)
            cache = QuantizedKVCache(bits=4, group_size=64)
            source = mx.arange(4*128).reshape(1, 1, 4, 128).astype(mx.bfloat16)
            cache.update_and_fetch(source, source)
            mx.eval(cache.state)
            disk.insert_cache(('model', None, None), [1,2,3,4,5], [cache])
            self.assertEqual(disk.nbytes, 0)
            self.assertEqual(len(disk), 1)
            disk = DiskPromptCache(directory, 'test', max_bytes=10**7)
            loaded, rest = disk.fetch_nearest_cache(('model',None,None), [1,2,3,9])
            self.assertEqual(rest, [9])
            self.assertEqual(loaded[0].offset, 3)
            self.assertEqual(loaded[0].bits, 4)
            self.assertTrue(mx.array_equal(loaded[0].keys[0], cache.keys[0][...,:3,:]))
            loaded[0].update_and_fetch(source[...,:1,:], source[...,:1,:])
            mx.eval(loaded[0].state)
            self.assertEqual(loaded[0].offset, 4)
            exact, tail = disk.fetch_nearest_cache(('model',None,None), [1,2,3,4])
            self.assertEqual(exact[0].offset, 3)
            self.assertEqual(tail, [4])
            miss, tail = disk.fetch_nearest_cache(('different',None,None), [1,2,3,4])
            self.assertIsNone(miss)
            disk.restore_budget = 0
            self.assertEqual(disk.fetch_nearest_cache(('model',None,None), [1,2,3,4]),
                             (None, [1,2,3,4]))
            disk.max_bytes = 0
            disk._evict(0)
            self.assertEqual(len(disk), 0)
            self.assertFalse(list(Path(directory).rglob('*.safetensors')))

    def test_corrupt_cache_falls_back(self):
        import mlx.core as mx
        from mlx_lm.models.cache import QuantizedKVCache
        from disk_cache import DiskPromptCache
        with tempfile.TemporaryDirectory() as directory:
            disk = DiskPromptCache(directory, 'test')
            cache = QuantizedKVCache(bits=4)
            x = mx.zeros((1,1,2,128), dtype=mx.bfloat16)
            cache.update_and_fetch(x,x)
            disk.insert_cache(('model',None,None), [1,2], [cache])
            next(Path(directory).rglob('*.safetensors')).write_bytes(b'broken')
            self.assertEqual(disk.fetch_nearest_cache(('model',None,None), [1,2,3]), (None,[1,2,3]))


if __name__ == '__main__':
    unittest.main()


class CacheCapabilityTest(unittest.TestCase):
    """The serving flags depend on cache behaviour, so it is probed, not assumed."""

    def test_capabilities_match_the_cache_types_the_runtime_ships(self):
        from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
        from runtime_support import cache_capabilities

        plain = cache_capabilities([KVCache()])
        self.assertTrue(plain['quantize_safe'] and plain['quantizes_any'] and plain['mergeable'])

        # A hybrid model quantizes its attention caches and leaves recurrent
        # state alone, which upstream's maybe_quantize_kv_cache already allows.
        hybrid = cache_capabilities([KVCache(), ArraysCache(2)])
        self.assertTrue(hybrid['quantize_safe'] and hybrid['quantizes_any'])

        # The rotating cache offers to_quantized and raises, so accepting the
        # flag would take the server down mid-request.
        rotating = cache_capabilities([RotatingKVCache(max_size=128)])
        self.assertFalse(rotating['quantize_safe'])
        self.assertFalse(rotating['quantizes_any'])

        # A model whose only caches are recurrent gains nothing from the flag.
        recurrent = cache_capabilities([ArraysCache(2)])
        self.assertTrue(recurrent['quantize_safe'])
        self.assertFalse(recurrent['quantizes_any'])

    def test_no_caches_is_not_reported_as_capable(self):
        from runtime_support import cache_capabilities
        self.assertEqual(cache_capabilities([]), {'quantize_safe': True, 'quantizes_any': False,
                                                  'mergeable': False})
