"""Lazy Windows RSS + whole-GPU guard. Import never initializes CUDA.

Cooperative checks supplement an independent process supervisor; they cannot
preempt an already executing kernel at an exact byte limit.
"""
from pathlib import Path
import shutil
import subprocess
import time

from .contracts import ContractError


class SplitExecutionGuard:
    def __init__(self, authorization, *, stop_file=None, min_free_ram=8*1024**3,
                 min_free_disk=8*1024**3, output_directory=None):
        from .background_checkpoint import MemoryGuard
        self.authorization = dict(authorization)
        self.memory = MemoryGuard(authorization['max_rss_bytes'], min_free_ram)
        self.stop_file = None if stop_file is None else Path(stop_file)
        self.output = None if output_directory is None else Path(output_directory)
        self.min_free_disk = int(min_free_disk)
        self.last = {}
        self.stop_reason = None
        self._next_poll = 0.
        self._global_peak = 0
        self._previous = []

    def check(self, *, force=False, chunk_boundary=False):
        a = self.authorization
        if self.stop_reason:
            return False
        now = time.time()
        from .useful_runtime_budget import goal_only_authorization
        goal_only=goal_only_authorization(a)
        if not goal_only and now >= min(a['expires_unix'], a['round_started_unix']+a['max_wall_seconds']):
            self.stop_reason = 'runtime_permit_or_total_wall_expired'
        elif self.stop_file is not None and self.stop_file.exists():
            self.stop_reason = 'explicit_stop_file'
        if self.stop_reason:
            return False
        if force or time.monotonic() >= self._next_poll:
            self._next_poll = time.monotonic()+2.
            if not self.memory.check():
                self.stop_reason = self.memory.stop_reason
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    timeout=3, check=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                rows = result.stdout.strip().splitlines()
                if len(rows) != 1:
                    raise ValueError('Expected one explicitly selected home GPU')
                used = int(rows[0].strip())*1024**2
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.stop_reason = 'whole_gpu_memory_measurement_unavailable'
                self.last['measurement_error'] = type(exc).__name__
                return False
            self._global_peak = max(self._global_peak, used)
            self.last.update(self.memory.last, global_gpu_used_bytes=used,
                             sampled_peak_global_gpu_bytes=self._global_peak)
            if used > a['max_vram_bytes']:
                self.stop_reason = 'whole_gpu_memory_limit'
            if self.output is not None:
                free = shutil.disk_usage(self.output).free
                self.last['free_disk_bytes'] = free
                if free < self.min_free_disk:
                    self.stop_reason = 'disk_reserve'
        if chunk_boundary:
            self._previous.append(int(self.memory.last.get('rss_bytes', 0)))
            self._previous = self._previous[-6:]
            if len(self._previous) == 6 and all(y-x > 32*1024**2 for x,y in
                    zip(self._previous, self._previous[1:])) and self._previous[-1]-self._previous[0] > 512*1024**2:
                self.stop_reason = 'sustained_chunk_boundary_rss_growth'
        self.last['stop_reason'] = self.stop_reason
        return self.stop_reason is None

    def require(self, **kwargs):
        if not self.check(**kwargs):
            raise ContractError('Split execution safety stop: '+str(self.stop_reason))
