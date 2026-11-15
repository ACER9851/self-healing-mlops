"""
pipeline.py — Orchestrator connecting buffer → drift → adversarial → RL → retrain → validate → deploy.

- asyncio.Lock for retraining guard (Fix #11: prevent concurrent retrain triggers).
- Async retraining so /api/predict isn't blocked (Fix #10).
- Post-deployment settling window (Fix #13).

Step 1: Stub orchestrator with function signatures.
Full implementation wired in Steps 3–7.
"""

import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import SETTLING_WINDOW_CYCLES


class Pipeline:
    """
    The self-healing pipeline orchestrator.

    Connects all 11 components in sequence:
    buffer_full → drift_check → adversarial_test → rl_decide →
    retrain → validate → deploy_or_ask_human → llm_report
    """

    def __init__(self):
        # Fix #11: pipeline-level retraining lock
        self._retrain_lock = asyncio.Lock()
        self._is_retraining = False

        # Current model version tracking
        self._current_model_version: Optional[str] = None

        # Fix #13: settling window tracking
        self._settling_cycles_remaining = 0

        # WebSocket broadcast function (set by server.py)
        self._ws_broadcast = None

    def set_ws_broadcast(self, broadcast_fn):
        """Set the WebSocket broadcast function for live dashboard updates."""
        self._ws_broadcast = broadcast_fn

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Send a live update to the dashboard via WebSocket."""
        if self._ws_broadcast:
            await self._ws_broadcast({
                "type": event_type,
                "data": data,
            })

    async def on_buffer_full(self, batch: Dict[str, Any]):
        """
        Called by buffer.py when the tumbling window fills up.
        This is the main pipeline trigger.

        Step 1: logs the event. Steps 3–7 will wire in the full pipeline.
        """
        from app2_mlops.backend.database import log_event

        cycle = batch.get("cycle", 0)
        size = batch.get("size", 0)

        # Fix #13: check settling window
        if self._settling_cycles_remaining > 0:
            self._settling_cycles_remaining -= 1
            await log_event(
                "SETTLING",
                f"Buffer cycle {cycle} skipped (settling window, {self._settling_cycles_remaining} remaining)",
                self._current_model_version,
            )
            await self.broadcast("settling", {
                "cycle": cycle,
                "remaining": self._settling_cycles_remaining,
            })
            return

        # Fix #11: don't start a new pipeline run if one is already in progress
        if self._is_retraining:
            await log_event(
                "PIPELINE_SKIP",
                f"Buffer cycle {cycle} skipped (retrain in progress)",
                self._current_model_version,
            )
            return

        await log_event(
            "BUFFER_FULL",
            f"Buffer cycle {cycle} complete with {size} samples — pipeline trigger",
            self._current_model_version,
        )
        await self.broadcast("buffer_full", {"cycle": cycle, "size": size})

        # Steps 3–7 will add: drift_check → adversarial → rl_decide → retrain → validate → deploy

    def notify_deployment(self):
        """Called after a successful deployment. Starts the settling window (Fix #13)."""
        self._settling_cycles_remaining = SETTLING_WINDOW_CYCLES


# Singleton instance
pipeline = Pipeline()
