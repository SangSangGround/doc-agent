"""Unit tests for disconnection/recovery without any real camera, TTS or network."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detection'))
from camera_recovery_v8 import CameraRecovery


class Frame:
    size = 100


class FakeCapture:
    def __init__(self, responses, opened=True):
        self.responses = list(responses)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        return self.responses.pop(0) if self.responses else (False, None)

    def release(self):
        self.released = True


class CameraRecoveryTests(unittest.TestCase):
    def test_initial_failure_does_not_exit_or_return_stale_frame(self):
        t = [0.0]
        cap = FakeCapture([(False, None), (True, Frame())])
        r = CameraRecovery(cap, lambda: FakeCapture([]), clock=lambda: t[0])
        frame, status = r.read()
        self.assertIsNone(frame)
        self.assertEqual(status, 'waiting')
        frame, status = r.read()
        self.assertIsNotNone(frame)
        self.assertEqual(status, 'first_frame')
        r.close()
        self.assertTrue(cap.released)

    def test_reopen_after_dead_camera_but_same_chosen_device(self):
        t = [0.0]
        first = FakeCapture([(False, None)])
        replacement = FakeCapture([(True, Frame())])
        opened = []
        def opener():
            opened.append('reopened')
            return replacement
        r = CameraRecovery(first, opener, clock=lambda: t[0], retry_interval=2.5)
        self.assertIsNone(r.read()[0])
        t[0] = 2.6
        frame, status = r.read()
        self.assertIsNone(frame)
        self.assertEqual(status, 'retry_open')
        self.assertTrue(first.released)
        self.assertEqual(opened, ['reopened'])
        frame, status = r.read()
        self.assertIsNotNone(frame)
        self.assertEqual(status, 'first_frame')

    def test_loss_then_recovery_invalidates_frame(self):
        t = [0.0]
        cap = FakeCapture([(True, Frame()), (False, None), (True, Frame())])
        r = CameraRecovery(cap, lambda: None, clock=lambda: t[0])
        self.assertEqual(r.read()[1], 'first_frame')
        frame, status = r.read()
        self.assertIsNone(frame)
        self.assertEqual(status, 'lost')
        self.assertEqual(r.read()[1], 'recovered')

    def test_unopened_camera_reconnects(self):
        t = [0.0]
        first = FakeCapture([], opened=False)
        second = FakeCapture([(True, Frame())])
        r = CameraRecovery(first, lambda: second, retry_interval=1, clock=lambda: t[0])
        self.assertEqual(r.read()[1], 'waiting')
        t[0] = 1.1
        self.assertEqual(r.read()[1], 'retry_open')
        self.assertEqual(r.read()[1], 'first_frame')

    def test_opener_exception_fails_closed_and_retries(self):
        t = [0.0]
        first = FakeCapture([])
        r = CameraRecovery(first, lambda: (_ for _ in ()).throw(RuntimeError('Camo busy')),
                           retry_interval=1, clock=lambda: t[0])
        t[0] = 1.1
        frame, status = r.read()
        self.assertIsNone(frame)
        self.assertEqual(status, 'retry_open')
        self.assertIn('Camo busy', r.last_error)
        self.assertIsNone(r.read()[0])


if __name__ == '__main__':
    unittest.main()
