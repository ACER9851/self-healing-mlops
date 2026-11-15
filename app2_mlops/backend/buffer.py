"""
buffer.py — Tumbling-window buffer for prediction samples (Component 2).

- Fixed-size tumbling window (default 500 samples).
- Fills to BUFFER_SIZE, triggers drift check callback, resets to empty.
- All access guarded by asyncio.Lock (Fix #12: thread-safety of shared state).
- NOT a sliding/circular buffer — discrete batch boundaries required by
  Fix #1 (statistical tests assume stable sample size) and Fix #6/#13
  (discrete buffer cycles for debouncing and settling windows).
"""

import asyncio
import numpy as np
from typing import Callable, Optional, List, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import BUFFER_SIZE


class PredictionBuffer:
    """
    Tumbling window that accumulates samples and triggers a callback when full.

    Usage:
        buffer = PredictionBuffer(on_buffer_full=my_drift_check_callback)
        await buffer.add_sample(features, prediction, request_id)
    """

    def __init__(
        self,
        buffer_size: int = BUFFER_SIZE,
        on_buffer_full: Optional[Callable] = None,
    ):
        self._buffer_size = buffer_size
        self._on_buffer_full = on_buffer_full
        self._lock = asyncio.Lock()

        # Internal storage
        self._features: List[np.ndarray] = []
        self._predictions: List[int] = []
        self._true_labels: List[Optional[int]] = []
        self._request_ids: List[str] = []

        # Cycle tracking (Fix #13: settling window, Fix #6: debouncing)
        self._cycle_count = 0
        self._cycles_since_deployment = 0  # Fix #13: skip drift detection post-deploy

    async def add_sample(
        self,
        features: np.ndarray,
        prediction: int,
        request_id: str,
        true_label: Optional[int] = None,
    ) -> bool:
        """
        Add a sample to the buffer. Returns True if buffer became full and
        the drift-check callback was triggered.

        Thread-safe via asyncio.Lock (Fix #12).
        """
        triggered = False

        async with self._lock:
            # Only accept if we haven't hit the size limit yet (tumbling, not sliding)
            if len(self._features) >= self._buffer_size:
                # Buffer is full but hasn't been processed yet — drop sample
                # This shouldn't happen if the callback resets promptly,
                # but guards against unbounded growth
                return False

            self._features.append(np.asarray(features, dtype=np.float32))
            self._predictions.append(int(prediction))
            self._true_labels.append(true_label)
            self._request_ids.append(request_id)

            if len(self._features) >= self._buffer_size:
                triggered = True

        # Trigger callback OUTSIDE the lock to avoid holding it during
        # potentially long drift detection / pipeline work
        if triggered and self._on_buffer_full is not None:
            batch = await self.get_current_batch()
            await self._on_buffer_full(batch)
            await self.reset()

        return triggered

    async def add_true_label(self, request_id: str, true_label: int) -> bool:
        """
        Attach a delayed true label to an existing buffered sample.
        Returns True if found and updated, False otherwise.
        """
        async with self._lock:
            for i, rid in enumerate(self._request_ids):
                if rid == request_id:
                    self._true_labels[i] = true_label
                    return True
        return False

    async def get_current_batch(self) -> Dict[str, Any]:
        """Return a snapshot of the current buffer contents."""
        async with self._lock:
            return {
                "features": np.array(self._features) if self._features else np.array([]),
                "predictions": np.array(self._predictions) if self._predictions else np.array([]),
                "true_labels": self._true_labels.copy(),
                "request_ids": self._request_ids.copy(),
                "size": len(self._features),
                "cycle": self._cycle_count,
            }

    async def reset(self):
        """Clear the buffer and increment the cycle counter (tumbling window)."""
        async with self._lock:
            self._features.clear()
            self._predictions.clear()
            self._true_labels.clear()
            self._request_ids.clear()
            self._cycle_count += 1

    async def get_size(self) -> int:
        """Current number of samples in buffer."""
        async with self._lock:
            return len(self._features)

    async def get_cycle_count(self) -> int:
        """Total number of completed buffer cycles."""
        async with self._lock:
            return self._cycle_count

    # ──────────────────────────────────────────
    # Fix #13: Post-deployment settling window
    # ──────────────────────────────────────────
    def notify_deployment(self):
        """Called after a successful deployment to start the settling window."""
        self._cycles_since_deployment = 0

    async def should_skip_drift_check(self, settling_cycles: int = 1) -> bool:
        """
        Fix #13: Returns True if we're within the settling window after deployment.
        Drift detection should be skipped/discounted for this many buffer cycles.
        """
        async with self._lock:
            if self._cycles_since_deployment < settling_cycles:
                self._cycles_since_deployment += 1
                return True
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Return buffer status for the dashboard."""
        async with self._lock:
            return {
                "current_size": len(self._features),
                "buffer_size": self._buffer_size,
                "fill_percent": round(len(self._features) / self._buffer_size * 100, 1),
                "cycle_count": self._cycle_count,
                "cycles_since_deployment": self._cycles_since_deployment,
            }
