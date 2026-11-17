# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Periodic progress checks for the local SFT experiment controller."""
import datetime
import json
import math
from pathlib import Path
import re
import time


class TrainingHealthMonitor:
    """Observe live launcher handles and bounded log tails without touching training."""

    def __init__(self, evidence: Path, phase: str, interval_seconds: int = 600):
        self.evidence = evidence
        self.phase = phase
        self.interval_seconds = interval_seconds
        self.last_step = 0
        self.last_progress_at = time.monotonic()

    def check(self, processes):
        """Append a health record; warnings require investigation, not automatic restart."""
        tails = []
        for path in sorted(self.evidence.glob(self.phase + '_node_*.log')):
            with path.open('rb') as handle:
                handle.seek(max(0, path.stat().st_size - 262144))
                tails.append(handle.read().decode(errors='replace'))
        text = '\n'.join(tails)
        metrics = []
        for line in text.splitlines():
            match = re.search(r'iteration\s+(\d+)/\s*(\d+)\s*\|', line)
            if match:
                values = {name: float(value) for name, value in re.findall(
                    r'(lm loss|mtp_1 loss|grad norm|learning rate|number of skipped iterations|number of nan iterations):\s*([-+]?(?:nan|inf|\d[\d.eE+\-]*))', line,
                    flags=re.I)}
                metrics.append({'step': int(match[1]), 'total_steps': int(match[2]), **values})
        latest = max(metrics, key=lambda row: row['step']) if metrics else {}
        step = latest.get('step', 0)
        now = time.monotonic()
        if step > self.last_step:
            self.last_step = step
            self.last_progress_at = now
        warnings = []
        if any(not math.isfinite(value) for row in metrics for value in row.values()):
            warnings.append('nonfinite_training_metric')
        if any(value > 0 for row in metrics for key, value in row.items() if 'iterations' in key):
            warnings.append('skipped_or_nan_iterations')
        if re.search(r'Traceback|DistBackendError|ncclRemoteError|out of memory', text, re.I):
            warnings.append('error_in_log_tail')
        if any(process.returncode not in (None, 0) for process in processes.values()):
            warnings.append('launcher_failed')
        limit = 1200 if step else 1800
        if now - self.last_progress_at >= limit:
            warnings.append('no_step_progress_20min' if step else 'initialization_over_30min')
        record = {
            'time': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'phase': self.phase, 'status': 'attention_required' if warnings else ('running' if step else 'starting'),
            'latest': latest, 'seconds_since_step_advance': round(now - self.last_progress_at, 1),
            'launchers': {rank: {'pid': process.pid, 'returncode': process.returncode}
                          for rank, process in processes.items()}, 'warnings': warnings,
        }
        if processes and all(process.returncode == 0 for process in processes.values()) and not warnings:
            record['status'] = 'finished'
        with (self.evidence / 'health_checks.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        return record
