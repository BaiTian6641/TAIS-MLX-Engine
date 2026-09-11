"""The service layer signals processes, so its safety property is the important one:
a stale record must never get an unrelated pid killed."""
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

import services


class IsOurServerTest(unittest.TestCase):
    def test_a_pid_that_is_not_a_server_of_ours_is_refused(self):
        # This test process is a python, but it is not `serve.py` under the repo.
        self.assertFalse(services.is_our_server(os.getpid()))

    def test_a_dead_pid_is_not_alive(self):
        self.assertFalse(services.is_our_server(2 ** 22 + 12345))

    def test_no_pid_is_not_alive(self):
        self.assertFalse(services.is_our_server(None))
        self.assertFalse(services.is_our_server(0))


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original = services.RUN
        services.RUN = Path(self.tmp.name)

    def tearDown(self):
        services.RUN = self.original
        self.tmp.cleanup()

    def write(self, port, **fields):
        record = {'pid': os.getpid(), 'alias': 'toy', 'port': port,
                  'started': 0, **fields}
        services.record_path(port).write_text(json.dumps(record))
        return record

    def test_load_reports_liveness_for_a_pid_that_is_not_ours(self):
        self.write(8080)
        record = services.load(8080)
        self.assertIsNotNone(record)
        self.assertFalse(record['alive'])  # the test process is not serve.py

    def test_stop_refuses_a_pid_that_is_not_ours(self):
        self.write(8080)
        result = services.stop(8080)
        self.assertFalse(result['stopped'])
        self.assertIn('not one of our servers', result['reason'])

    def test_stop_on_an_unknown_port_reports_so(self):
        result = services.stop(8099)
        self.assertFalse(result['stopped'])
        self.assertIn('no record', result['reason'])

    def test_forget_removes_the_record_and_its_metrics(self):
        self.write(8080)
        services.metrics_path(8080).write_text('{}')
        services.forget(8080)
        self.assertFalse(services.record_path(8080).exists())
        self.assertFalse(services.metrics_path(8080).exists())

    def test_instances_lists_records_and_skips_metrics_files(self):
        self.write(8080)
        self.write(8081, alias='other')
        services.metrics_path(8081).write_text('{}')
        found = {record['port'] for record in services.instances(include_dead=True)}
        self.assertEqual(found, {8080, 8081})

    def test_a_corrupt_record_is_ignored_rather_than_raising(self):
        services.record_path(8080).write_text('{not json')
        self.assertIsNone(services.load(8080))
        self.assertEqual(services.instances(), [])

    def test_metrics_reads_the_per_port_file(self):
        services.metrics_path(8080).write_text(json.dumps({'decode_tps': 42.0}))
        self.assertEqual(services.metrics(8080), {'decode_tps': 42.0})

    def test_metrics_returns_none_when_absent(self):
        self.assertIsNone(services.metrics(8099))

    def test_tail_returns_the_last_lines(self):
        services.log_path(8080).write_text('\n'.join(f'line {n}' for n in range(100)))
        self.assertEqual(services.tail(8080, 3), ['line 97', 'line 98', 'line 99'])

    def test_summary_mentions_the_alias_and_port(self):
        self.write(8080)
        self.assertIn('toy on port 8080', services.summary(8080))


class FreePortTest(unittest.TestCase):
    def test_a_bound_port_is_skipped(self):
        import console

        with socket.socket() as held:
            held.bind(('127.0.0.1', 0))
            held.listen(1)
            busy = held.getsockname()[1]
            self.assertNotEqual(console.free_port(busy), busy)


class CompatInstallTest(unittest.TestCase):
    def test_install_patches_both_verbs_and_leaves_them_working(self):
        from mlx_lm import server
        import llamacpp_compat

        before_get, before_post = server.APIHandler.do_GET, server.APIHandler.do_POST
        try:
            llamacpp_compat.install(server, {'alias': 'toy', 'path': '.', 'config': {}},
                                    type('O', (), {'port': 8080, 'mtp_draft': None})())
            self.assertIsNot(server.APIHandler.do_GET, before_get)
            self.assertIsNot(server.APIHandler.do_POST, before_post)
        finally:
            server.APIHandler.do_GET, server.APIHandler.do_POST = before_get, before_post
