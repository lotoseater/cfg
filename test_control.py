"""Verify refusal and restoration ordering without signalling any real process."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import control


class Lifecycle(unittest.TestCase):
    def fixture(self, root):
        return SimpleNamespace(pid=123,control_root=root/'control',gpus='0')

    def test_other_gpu_job_prevents_signal(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(control,'BASE',root),patch.object(control,'bounded',lambda p:Path(p)), \
                 patch.object(control,'snapshot',return_value={'pid':123}), \
                 patch.object(control,'active_pids',return_value={123,999}),patch.object(control.os,'kill') as kill:
                (root/'HDD/cfg').mkdir(parents=True)
                with self.assertRaises(RuntimeError):
                    control.main(self.fixture(root))
                kill.assert_not_called()

    def test_pid_change_prevents_signal(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(control,'BASE',root),patch.object(control,'bounded',lambda p:Path(p)), \
                 patch.object(control,'snapshot',side_effect=[{'pid':123,'start_ticks':'1'}, {'pid':123,'start_ticks':'2'}]), \
                 patch.object(control,'active_pids',return_value={123}),patch.object(control.os,'kill') as kill:
                (root/'HDD/cfg').mkdir(parents=True)
                with self.assertRaises(RuntimeError):
                    control.main(self.fixture(root))
                kill.assert_not_called()

    def exercise(self, formal_fails=False):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'HDD/cfg').mkdir(parents=True)
            args=self.fixture(root)
            events=[]
            def child(control_root,mode,gpus,smoke=None):
                events.append(mode)
                if mode=='smoke':
                    path=root/'smoke'
                    (control_root/'proceed.json').write_text(json.dumps({'smoke_root':str(path)}))
                    return path
                if formal_fails:
                    raise RuntimeError('simulated formal failure')
                return root/'formal'
            def restore(control_root,meta):
                events.append('restore')
                return 456
            with patch.object(control,'BASE',root),patch.object(control,'bounded',lambda p:Path(p)), \
                 patch.object(control,'snapshot',return_value={'pid':123,'start_ticks':'1'}), \
                 patch.object(control,'active_pids',side_effect=[{123},set(),set()]), \
                 patch.object(control,'original_alive',return_value=False), \
                 patch.object(control.os,'kill',side_effect=lambda *args:events.append('stop')), \
                 patch.object(control,'run_child',side_effect=child),patch.object(control,'restore',side_effect=restore), \
                 patch.object(control.subprocess,'Popen') as replay:
                replay.return_value.wait.return_value=0
                if formal_fails:
                    with self.assertRaises(RuntimeError):
                        control.main(args)
                else:
                    control.main(args)
            self.assertEqual(events,['stop','smoke','formal','restore'])
            expected='FAILED_RESTORED' if formal_fails else 'COMPLETE_RESTORED'
            self.assertEqual(json.loads((args.control_root/'state.json').read_text())['status'],expected)

    def test_restore_after_all_work(self):
        self.exercise()

    def test_restore_after_failure_without_other_signals(self):
        self.exercise(True)


if __name__=='__main__':
    unittest.main(verbosity=2)
