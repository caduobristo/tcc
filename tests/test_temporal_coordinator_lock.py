"""Regression checks for a reused Windows PID and competing probe processes."""
from pathlib import Path
from unittest import TestCase, mock
import psutil

from scripts.run_temporal_interpolation import coordinator_alive

SCRIPT = str(Path(__file__).resolve().parents[1]/'scripts/run_temporal_interpolation.py')


class CoordinatorLockTests(TestCase):
    def test_unrelated_reused_pid_is_not_a_coordinator(self):
        with mock.patch('psutil.Process') as process:
            process.return_value.cmdline.return_value = ['conhost.exe', '0x4']
            self.assertFalse(coordinator_alive({'pid': 2156}))

    def test_requires_correct_command_and_process_creation_time(self):
        with mock.patch('psutil.Process') as process:
            process.return_value.cmdline.return_value = ['python.exe', SCRIPT, '--run']
            process.return_value.create_time.return_value = 100.0
            self.assertTrue(coordinator_alive({'pid': 42, 'process_created_at': 100.0}))
            self.assertFalse(coordinator_alive({'pid': 42, 'process_created_at': 99.0}))
            process.return_value.cmdline.return_value = ['python.exe', SCRIPT, '--worker', 'extract']
            self.assertFalse(coordinator_alive({'pid': 42}))

    def test_missing_pid_is_stale_but_unreadable_pid_is_not_assumed_stale(self):
        with mock.patch('psutil.Process', side_effect=psutil.NoSuchProcess(42)):
            self.assertFalse(coordinator_alive({'pid': 42}))
        with mock.patch('psutil.Process', side_effect=psutil.AccessDenied(42)):
            with self.assertRaises(RuntimeError):
                coordinator_alive({'pid': 42})
