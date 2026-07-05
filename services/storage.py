from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                kpi TEXT NOT NULL,
                constraints_text TEXT,
                language TEXT NOT NULL,
                knowledge_bases_json TEXT NOT NULL DEFAULT '[]',
                documents_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                rationale TEXT,
                mechanism TEXT,
                kpi_link TEXT,
                constraints_fit TEXT,
                expected_effect TEXT,
                industrial_scale TEXT,
                novelty_score REAL,
                novelty_why TEXT,
                risk_score REAL,
                risk_why TEXT,
                value_score REAL,
                value_why TEXT,
                economic_value_score REAL,
                economic_value_why TEXT,
                success_probability_score REAL,
                success_probability_why TEXT,
                verification_recommendation TEXT,
                final_score REAL,
                status TEXT,
                is_verified INTEGER DEFAULT 0,
                expert_rating INTEGER,
                expert_comment TEXT,
                verified_at TEXT,
                evidence_json TEXT,
                roadmap_json TEXT,
                causal_chain_json TEXT,
                resource_estimate_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                comment TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

        run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        if "knowledge_bases_json" not in run_columns:
            conn.execute(
                "ALTER TABLE runs ADD COLUMN knowledge_bases_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "documents_json" not in run_columns:
            conn.execute(
                "ALTER TABLE runs ADD COLUMN documents_json TEXT NOT NULL DEFAULT '[]'"
            )

        hypothesis_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(hypotheses)")
        }
        if "success_probability_score" not in hypothesis_columns:
            conn.execute(
                "ALTER TABLE hypotheses ADD COLUMN success_probability_score REAL"
            )
        if "success_probability_why" not in hypothesis_columns:
            conn.execute(
                "ALTER TABLE hypotheses ADD COLUMN success_probability_why TEXT"
            )
        if "uncertainty_score" in hypothesis_columns:
            conn.execute(
                """
                UPDATE hypotheses
                SET success_probability_score = 1.0 - uncertainty_score
                WHERE success_probability_score IS NULL
                  AND uncertainty_score IS NOT NULL
                """
            )
        if "uncertainty_why" in hypothesis_columns:
            conn.execute(
                """
                UPDATE hypotheses
                SET success_probability_why = uncertainty_why
                WHERE (success_probability_why IS NULL OR success_probability_why = '')
                  AND uncertainty_why IS NOT NULL
                """
            )


def create_run(
    db_path: Path,
    *,
    kpi: str,
    constraints_text: str,
    language: str,
    knowledge_bases: list[str],
    documents: list[dict[str, Any]] | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (id, kpi, constraints_text, language, knowledge_bases_json, documents_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                kpi,
                constraints_text,
                language,
                json.dumps(knowledge_bases, ensure_ascii=False),
                json.dumps(documents or [], ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    return run_id


def get_run(db_path: Path, run_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["knowledge_bases"] = json.loads(result.pop("knowledge_bases_json", "[]") or "[]")
    result["documents"] = json.loads(result.pop("documents_json", "[]") or "[]")
    return result


def get_latest_run(db_path: Path) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["knowledge_bases"] = json.loads(result.pop("knowledge_bases_json", "[]") or "[]")
    result["documents"] = json.loads(result.pop("documents_json", "[]") or "[]")
    return result


def save_hypotheses(
    db_path: Path,
    run_id: str,
    hypotheses: list[dict[str, Any]],
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        for item in hypotheses:
            hypothesis_id = item.get("id") or str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO hypotheses (
                    id, run_id, hypothesis, rationale, mechanism, kpi_link,
                    constraints_fit, expected_effect, industrial_scale,
                    novelty_score, novelty_why, risk_score, risk_why,
                    value_score, value_why, economic_value_score,
                    economic_value_why, success_probability_score, success_probability_why,
                    verification_recommendation, final_score, status,
                    is_verified, expert_rating, expert_comment, verified_at,
                    evidence_json, roadmap_json, causal_chain_json,
                    resource_estimate_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id,
                    run_id,
                    item["hypothesis"],
                    item.get("rationale", ""),
                    item.get("mechanism", ""),
                    item.get("kpi_link", ""),
                    item.get("constraints_fit", ""),
                    item.get("expected_effect", ""),
                    item.get("industrial_scale", ""),
                    float(item.get("novelty", {}).get("score", 0.0)),
                    item.get("novelty", {}).get("why", ""),
                    float(item.get("risk", {}).get("score", 0.0)),
                    item.get("risk", {}).get("why", ""),
                    float(item.get("value", {}).get("score", 0.0)),
                    item.get("value", {}).get("why", ""),
                    float(item.get("economic_value", {}).get("score", 0.0)),
                    item.get("economic_value", {}).get("why", ""),
                    float(item.get("success_probability", {}).get("score", 0.0)),
                    item.get("success_probability", {}).get("why", ""),
                    item.get("verification_recommendation", ""),
                    float(item.get("final_score", 0.0)),
                    item.get("status", "draft"),
                    int(bool(item.get("is_verified", False))),
                    item.get("expert_rating"),
                    item.get("expert_comment", ""),
                    item.get("verified_at"),
                    json.dumps(item.get("evidence", []), ensure_ascii=False),
                    json.dumps(item.get("roadmap", []), ensure_ascii=False),
                    json.dumps(item.get("causal_chain", []), ensure_ascii=False),
                    json.dumps(item.get("resource_estimate", {}), ensure_ascii=False),
                    now,
                    now,
                ),
            )


def get_hypotheses(db_path: Path, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM hypotheses WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["is_verified"] = bool(item["is_verified"])
        item["evidence"] = json.loads(item.pop("evidence_json") or "[]")
        item["roadmap"] = json.loads(item.pop("roadmap_json") or "[]")
        item["causal_chain"] = json.loads(item.pop("causal_chain_json") or "[]")
        item["resource_estimate"] = json.loads(item.pop("resource_estimate_json") or "{}")
        if item.get("success_probability_score") is None:
            legacy = item.get("uncertainty_score")
            item["success_probability_score"] = (1.0 - float(legacy)) if legacy is not None else 0.0
        if not item.get("success_probability_why"):
            item["success_probability_why"] = item.get("uncertainty_why", "")
        result.append(item)
    return result


def update_status(
    db_path: Path,
    hypothesis_id: str,
    *,
    status: str,
    final_score: float,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE hypotheses
            SET status = ?, final_score = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                final_score,
                datetime.now().isoformat(timespec="seconds"),
                hypothesis_id,
            ),
        )


def save_expert_review(
    db_path: Path,
    hypothesis_id: str,
    *,
    is_verified: bool,
    expert_rating: int,
    expert_comment: str,
) -> None:
    verified_at = datetime.now().isoformat(timespec="seconds") if is_verified else None
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE hypotheses
            SET is_verified = ?, expert_rating = ?, expert_comment = ?,
                verified_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                int(is_verified),
                int(expert_rating),
                expert_comment,
                verified_at,
                datetime.now().isoformat(timespec="seconds"),
                hypothesis_id,
            ),
        )


def add_feedback(
    db_path: Path,
    *,
    run_id: str,
    hypothesis_id: str,
    action: str,
    old_value: str = "",
    new_value: str = "",
    comment: str = "",
) -> str:
    feedback_id = str(uuid.uuid4())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                id, run_id, hypothesis_id, action,
                old_value, new_value, comment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                run_id,
                hypothesis_id,
                action,
                old_value,
                new_value,
                comment,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    return feedback_id


def list_feedback(db_path: Path) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]
