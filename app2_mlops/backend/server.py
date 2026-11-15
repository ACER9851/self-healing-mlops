"""
server.py — FastAPI backend for the Self-Healing MLOps Engine (App 2, port 5000).

All endpoints from Part 6:
- /api/predict — runs inference via ModelWrapper, logs to buffer
- /api/observe — Fix #18: logs external predictions (no inference)
- /api/feedback — returns 404 for unknown request_id
- /api/status, /api/events, /api/drift, /api/accuracy, /api/gates,
  /api/models, /api/adversarial, /api/llm-report, /api/rl-state — dashboard reads
- /api/approve, /api/reject — auth-protected (Fix #16)
- /ws/dashboard — WebSocket for live updates with auto-reconnect support

CORS enabled for React dashboard.
Auth on all state-changing endpoints (Fix #16).
"""

import asyncio
import json
import uuid
import time
import numpy as np
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import (
    APP2_BACKEND_PORT, BUFFER_SIZE, API_KEY,
    DATASET_PROFILE, MODEL_TYPE, ADVERSARIAL_EPSILON,
)
from app2_mlops.backend.database import init_db, get_connection, db_write_lock, log_event
from app2_mlops.backend.buffer import PredictionBuffer
from app2_mlops.backend.pipeline import pipeline
from app2_mlops.backend.auth import require_api_key


# ──────────────────────────────────────────────
# WebSocket connection manager
# ──────────────────────────────────────────────
class ConnectionManager:
    """Manages active WebSocket connections for dashboard live updates."""

    def __init__(self):
        self._connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send a JSON message to all connected dashboards."""
        async with self._lock:
            dead = []
            for ws in self._connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._connections.remove(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()


# ──────────────────────────────────────────────
# Buffer + pipeline wiring
# ──────────────────────────────────────────────
prediction_buffer = PredictionBuffer(
    buffer_size=BUFFER_SIZE,
    on_buffer_full=pipeline.on_buffer_full,
)

# Give the pipeline access to WS broadcast
pipeline.set_ws_broadcast(ws_manager.broadcast)


# ──────────────────────────────────────────────
# App lifecycle
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, log start. Shutdown: log stop."""
    init_db()
    await log_event("SYSTEM", "App 2 MLOps Engine started", None)
    await ws_manager.broadcast({"type": "system", "data": {"status": "started"}})
    yield
    await log_event("SYSTEM", "App 2 MLOps Engine stopping", None)


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Self-Healing MLOps Engine",
    description="Framework-agnostic self-healing ML pipeline with drift detection, "
                "adversarial testing, RL-based decision making, and automated retraining.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dashboard on any port during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic request/response models
# ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: list = Field(..., description="Feature vector (list of numbers)")
    request_id: Optional[str] = Field(None, description="Optional custom request ID")

class ObserveRequest(BaseModel):
    features: list = Field(..., description="Feature vector from your own model")
    prediction: int = Field(..., description="Your model's prediction")
    request_id: Optional[str] = Field(None, description="Optional custom request ID")
    timestamp: Optional[str] = Field(None, description="When your model made this prediction")

class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="The request ID to attach the label to")
    true_label: int = Field(..., description="The ground-truth label")

class ApprovalRequest(BaseModel):
    model_version: str = Field(..., description="Version to approve for deployment")
    reason: Optional[str] = Field("", description="Optional reason")

class RejectionRequest(BaseModel):
    model_version: str = Field(..., description="Version to reject")
    reason: Optional[str] = Field("", description="Optional reason for rejection")


# ──────────────────────────────────────────────
# Prediction endpoint — App 2 owns the model
# ──────────────────────────────────────────────
@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    Run inference via ModelWrapper, log to prediction_log + buffer.
    Used by our own demo (App 1 → App 2).
    """
    request_id = req.request_id or str(uuid.uuid4())
    features = np.array(req.features, dtype=np.float32)

    # TODO (Step 2): Load actual model and run inference
    # For now, return a dummy prediction
    prediction = 0
    model_version = "v0_initial"

    # Log to database
    async with db_write_lock:
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO prediction_log
                   (request_id, features, prediction, model_version, source)
                   VALUES (?, ?, ?, ?, 'predict')""",
                (request_id, json.dumps(req.features), prediction, model_version),
            )
            conn.commit()
        finally:
            conn.close()

    # Add to buffer for drift detection
    await prediction_buffer.add_sample(
        features=features,
        prediction=prediction,
        request_id=request_id,
    )

    # Broadcast to dashboard
    buffer_status = await prediction_buffer.get_status()
    await ws_manager.broadcast({
        "type": "prediction",
        "data": {
            "request_id": request_id,
            "prediction": prediction,
            "model_version": model_version,
            "buffer": buffer_status,
        },
    })

    return {
        "request_id": request_id,
        "prediction": prediction,
        "model_version": model_version,
        "buffer_status": buffer_status,
    }


# ──────────────────────────────────────────────
# Observe endpoint — Fix #18: external model monitoring
# ──────────────────────────────────────────────
@app.post("/api/observe")
async def observe(req: ObserveRequest):
    """
    Lightweight ingestion for a user's own already-deployed model.
    App 2 does NOT run inference — only logs into buffer for drift/adversarial detection.
    """
    request_id = req.request_id or str(uuid.uuid4())
    features = np.array(req.features, dtype=np.float32)

    # Log to database with model_version='external' (sentinel row from Fix #18)
    async with db_write_lock:
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO prediction_log
                   (request_id, features, prediction, model_version, source)
                   VALUES (?, ?, ?, 'external', 'observe')""",
                (request_id, json.dumps(req.features), req.prediction),
            )
            conn.commit()
        finally:
            conn.close()

    # Add to buffer — same drift/adversarial detection path
    await prediction_buffer.add_sample(
        features=features,
        prediction=req.prediction,
        request_id=request_id,
    )

    buffer_status = await prediction_buffer.get_status()

    return {
        "request_id": request_id,
        "observed": True,
        "buffer_status": buffer_status,
    }


# ──────────────────────────────────────────────
# Feedback endpoint — delayed true labels
# ──────────────────────────────────────────────
@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """
    Attach a true label to a previously logged prediction.
    Returns 404 for unknown request_id — not a silent 200 (execution-verified requirement).
    """
    async with db_write_lock:
        conn = get_connection()
        try:
            cursor = conn.execute(
                """UPDATE prediction_log
                   SET true_label = ?, label_received_at = ?
                   WHERE request_id = ?""",
                (req.true_label, datetime.utcnow().isoformat(), req.request_id),
            )
            conn.commit()
            rows_affected = cursor.rowcount
        finally:
            conn.close()

    if rows_affected == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found with request_id '{req.request_id}'",
        )

    # Also update buffer if the sample is still there
    await prediction_buffer.add_true_label(req.request_id, req.true_label)

    return {"request_id": req.request_id, "label_recorded": True}


# ──────────────────────────────────────────────
# Dashboard read endpoints
# ──────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    """Overall system status for the dashboard."""
    buffer_status = await prediction_buffer.get_status()
    conn = get_connection()
    try:
        # Latest model
        model = conn.execute(
            "SELECT * FROM model_registry WHERE status='active' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        # Counts
        pred_count = conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
        drift_count = conn.execute("SELECT COUNT(*) FROM drift_event").fetchone()[0]
        retrain_count = conn.execute("SELECT COUNT(*) FROM retrain_event").fetchone()[0]

        # Latest event
        latest_event = conn.execute(
            "SELECT * FROM event_log ORDER BY logged_at DESC LIMIT 1"
        ).fetchone()

        return {
            "status": "running",
            "dataset_profile": DATASET_PROFILE,
            "model_type": MODEL_TYPE,
            "current_model": dict(model) if model else None,
            "buffer": buffer_status,
            "counts": {
                "predictions": pred_count,
                "drift_events": drift_count,
                "retrains": retrain_count,
            },
            "latest_event": dict(latest_event) if latest_event else None,
            "ws_connections": ws_manager.count,
        }
    finally:
        conn.close()


@app.get("/api/events")
async def get_events(limit: int = Query(50, ge=1, le=500)):
    """Recent events from the event log."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM event_log ORDER BY logged_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"events": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/drift")
async def get_drift(limit: int = Query(50, ge=1, le=500)):
    """Drift detection history."""
    conn = get_connection()
    try:
        events = conn.execute(
            "SELECT * FROM drift_event ORDER BY detected_at DESC LIMIT ?", (limit,)
        ).fetchall()
        history = conn.execute(
            "SELECT * FROM drift_history ORDER BY recorded_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return {
            "drift_events": [dict(r) for r in events],
            "drift_history": [dict(r) for r in history],
        }
    finally:
        conn.close()


@app.get("/api/accuracy")
async def get_accuracy(limit: int = Query(50, ge=1, le=500)):
    """Accuracy/F1 history over time."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM accuracy_history ORDER BY recorded_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"accuracy_history": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/gates")
async def get_gates(limit: int = Query(20, ge=1, le=100)):
    """Validation gate results history."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM validation_result ORDER BY validated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"gate_results": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/models")
async def get_models():
    """Model registry — all model versions."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM model_registry WHERE version != 'external' ORDER BY created_at DESC"
        ).fetchall()
        return {"models": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/adversarial")
async def get_adversarial(limit: int = Query(20, ge=1, le=100)):
    """Adversarial test results history."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM adversarial_result ORDER BY tested_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"adversarial_results": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/llm-report")
async def get_llm_report():
    """Latest LLM diagnostic report."""
    # Step 7: will pull from llm_report.py
    return {
        "report": "LLM diagnostic reports will be generated starting Step 7.",
        "generated_at": None,
    }


@app.get("/api/rl-state")
async def get_rl_state():
    """Current Q-table state and RL decision history."""
    conn = get_connection()
    try:
        qtable = conn.execute("SELECT * FROM qtable_state ORDER BY state_id").fetchall()
        decisions = conn.execute(
            "SELECT * FROM rl_decision ORDER BY decided_at DESC LIMIT 20"
        ).fetchall()
        return {
            "qtable": [dict(r) for r in qtable],
            "recent_decisions": [dict(r) for r in decisions],
        }
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Auth-protected state-changing endpoints (Fix #16)
# ──────────────────────────────────────────────
@app.post("/api/approve", dependencies=[Depends(require_api_key)])
async def approve(req: ApprovalRequest):
    """Human approves deployment of a candidate model."""
    await log_event("HUMAN_APPROVE", f"Model {req.model_version} approved: {req.reason}", req.model_version)
    await ws_manager.broadcast({
        "type": "human_action",
        "data": {"action": "approve", "model_version": req.model_version, "reason": req.reason},
    })
    # Step 7: wire to deployment.approve_deployment()
    return {"approved": True, "model_version": req.model_version}


@app.post("/api/reject", dependencies=[Depends(require_api_key)])
async def reject(req: RejectionRequest):
    """Human rejects deployment of a candidate model."""
    await log_event("HUMAN_REJECT", f"Model {req.model_version} rejected: {req.reason}", req.model_version)
    await ws_manager.broadcast({
        "type": "human_action",
        "data": {"action": "reject", "model_version": req.model_version, "reason": req.reason},
    })
    # Step 7: wire to deployment.reject_deployment()
    return {"rejected": True, "model_version": req.model_version}


# ──────────────────────────────────────────────
# WebSocket endpoint for live dashboard updates
# ──────────────────────────────────────────────
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    WebSocket endpoint pushing live updates to the React dashboard.
    The dashboard's useWebSocket hook auto-reconnects on disconnect.
    """
    await ws_manager.connect(websocket)
    await log_event("WS_CONNECT", f"Dashboard connected (total: {ws_manager.count})")

    try:
        # Send initial state on connect
        buffer_status = await prediction_buffer.get_status()
        await websocket.send_json({
            "type": "init",
            "data": {
                "status": "connected",
                "dataset_profile": DATASET_PROFILE,
                "model_type": MODEL_TYPE,
                "buffer": buffer_status,
            },
        })

        # Keep connection alive, listen for client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Handle ping/pong for keepalive
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat to detect dead connections
                try:
                    await websocket.send_json({"type": "heartbeat", "data": {}})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await ws_manager.disconnect(websocket)
        await log_event("WS_DISCONNECT", f"Dashboard disconnected (remaining: {ws_manager.count})")


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ──────────────────────────────────────────────
# Run with: uvicorn app2_mlops.backend.server:app --port 5000 --reload
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app2_mlops.backend.server:app",
        host="0.0.0.0",
        port=APP2_BACKEND_PORT,
        reload=True,
    )
