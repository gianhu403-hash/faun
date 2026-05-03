"""YDB implementation of IncidentRepository.

Makes incidents **persistent** across restarts (unlike the in-memory default).
Chat-to-incident mapping remains in-memory for simplicity during demo --
it is transient by nature (cleared when a ranger finishes workflow).
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from cloud.db.base import IncidentRepository
from cloud.db.incidents import Incident, VALID_TRANSITIONS
from cloud.notify.districts import get_district_name

logger = logging.getLogger(__name__)

# Chat-to-incident mapping: transient, in-memory (same as in-memory version).
_chat_to_incident: dict[int, str] = {}

# Fields that can be persisted in YDB (excludes binary/blob fields)
_YDB_PERSISTABLE = frozenset(
    {
        "status",
        "accepted_by_chat_id",
        "accepted_by_name",
        "accepted_at",
        "arrived_at",
        "response_time_min",
        "district",
        "ranger_report_raw",
        "ranger_report_legal",
        "resolution_details",
        "is_demo",
        "created_at",
        "protocol_pdf",
        # FAUN-38a B2: persist drone evidence + alert tracking across restarts
        "drone_photo_b64",
        "drone_comment",
        "ranger_photo_b64",
        "alert_message_ids",
    }
)

# Map Python field names to YDB column types for parameterized queries
_FIELD_TYPES: dict[str, str] = {
    "status": "Utf8",
    "accepted_by_chat_id": "Int64",
    "accepted_by_name": "Utf8",
    "accepted_at": "Double",
    "arrived_at": "Double",
    "response_time_min": "Double",
    "district": "Utf8",
    "ranger_report_raw": "Utf8",
    "ranger_report_legal": "Utf8",
    "resolution_details": "Utf8",
    "is_demo": "Bool",
    "created_at": "Double",
    "protocol_pdf": "String",
    # FAUN-38a B2: drone evidence + alert tracking
    "drone_photo_b64": "String",  # raw PNG bytes -> YDB String (binary)
    "drone_comment": "Utf8",
    "ranger_photo_b64": "String",
    "alert_message_ids": "Utf8",  # JSON-serialised dict[chat_id]=msg_id
}


class YDBIncidentRepository(IncidentRepository):
    """YDB-backed incident storage with in-memory chat mapping."""

    def __init__(self) -> None:
        from cloud.db.ydb_client import ensure_tables

        ensure_tables()  # non-blocking: runs in background thread

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def create_incident(
        self,
        audio_class: str,
        lat: float,
        lon: float,
        confidence: float,
        gating_level: str,
        is_demo: bool = False,
    ) -> Incident:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()
        incident_id = str(uuid.uuid4())
        now = time.time()
        district = get_district_name(lat, lon)

        def _ins(session):
            from cloud.db.ydb_client import execute_query

            execute_query(
                session,
                """
                DECLARE $id AS Utf8;
                DECLARE $cls AS Utf8;
                DECLARE $lat AS Double;
                DECLARE $lon AS Double;
                DECLARE $conf AS Double;
                DECLARE $gate AS Utf8;
                DECLARE $status AS Utf8;
                DECLARE $chat_id AS Int64;
                DECLARE $name AS Utf8;
                DECLARE $at AS Double;
                DECLARE $created_at AS Double;
                DECLARE $district AS Utf8;
                DECLARE $is_demo AS Bool;
                DECLARE $arrived_at AS Double;
                DECLARE $response_time_min AS Double;
                DECLARE $report_raw AS Utf8;
                DECLARE $report_legal AS Utf8;
                DECLARE $resolution AS Utf8;
                UPSERT INTO incidents (id, audio_class, lat, lon,
                    confidence, gating_level, status,
                    accepted_by_chat_id, accepted_by_name, accepted_at,
                    created_at, district, is_demo,
                    arrived_at, response_time_min,
                    ranger_report_raw, ranger_report_legal,
                    resolution_details)
                VALUES ($id, $cls, $lat, $lon,
                    $conf, $gate, $status,
                    $chat_id, $name, $at,
                    $created_at, $district, $is_demo,
                    $arrived_at, $response_time_min,
                    $report_raw, $report_legal,
                    $resolution)
                """,
                {
                    "$id": incident_id,
                    "$cls": audio_class,
                    "$lat": lat,
                    "$lon": lon,
                    "$conf": confidence,
                    "$gate": gating_level,
                    "$status": "pending",
                    "$chat_id": 0,
                    "$name": "",
                    "$at": 0.0,
                    "$created_at": now,
                    "$district": district,
                    "$is_demo": is_demo,
                    "$arrived_at": 0.0,
                    "$response_time_min": 0.0,
                    "$report_raw": "",
                    "$report_legal": "",
                    "$resolution": "",
                },
            )

        pool.retry_operation_sync(_ins)

        return Incident(
            id=incident_id,
            audio_class=audio_class,
            lat=lat,
            lon=lon,
            confidence=confidence,
            gating_level=gating_level,
            status="pending",
            created_at=now,
            district=district,
            is_demo=is_demo,
        )

    def update_status(self, incident_id: str, status: str) -> None:
        self.update_incident(incident_id, status=status)

    def update_incident(self, incident_id: str, **fields) -> None:
        """Update incident fields in YDB with state machine validation."""
        from cloud.db.ydb_client import get_pool

        if not fields:
            return

        # FAUN-38a B2: log non-persistable fields so silent drops are auditable.
        # Logged before state-machine validation so the warning fires on the
        # caller's intent, not on whether the incident currently exists.
        unknown = set(fields) - _YDB_PERSISTABLE
        if unknown:
            logger.warning(
                "update_incident dropping non-persistable fields: %s",
                sorted(unknown),
            )

        # State machine validation: check current status first if changing status
        new_status = fields.get("status")
        if new_status:
            current = self.get_incident(incident_id)
            if not current:
                return
            if new_status != current.status:
                allowed = VALID_TRANSITIONS.get(current.status, set())
                if new_status not in allowed:
                    return

        # Filter to persistable fields only
        ydb_fields = {k: v for k, v in fields.items() if k in _YDB_PERSISTABLE}
        if not ydb_fields:
            return

        pool = get_pool()

        # Build dynamic UPDATE statement with DECLARE block
        set_clauses = []
        params = {"$id": incident_id}
        declares = ["DECLARE $id AS Utf8;"]
        for field_name, value in ydb_fields.items():
            param_name = f"${field_name}"
            ydb_type = _FIELD_TYPES.get(field_name, "Utf8")
            set_clauses.append(f"{field_name} = {param_name}")
            declares.append(f"DECLARE {param_name} AS {ydb_type};")

            # FAUN-38a B2: alert_message_ids is dict in code, JSON in YDB.
            # Reject non-dict (None, list, etc) explicitly — silent coercion to
            # "" would be indistinguishable from "no alerts ever sent".
            if field_name == "alert_message_ids":
                if value is None:
                    value = "{}"  # explicit empty map, distinguishable from broken
                elif isinstance(value, dict):
                    value = json.dumps({str(k): v for k, v in value.items()})
                else:
                    raise TypeError(
                        f"alert_message_ids must be dict or None, got {type(value).__name__}"
                    )

            # Convert None to appropriate zero value for YDB
            if value is None:
                if ydb_type == "Double":
                    params[param_name] = 0.0
                elif ydb_type == "Int64":
                    params[param_name] = 0
                elif ydb_type == "Bool":
                    params[param_name] = False
                else:
                    params[param_name] = ""
            else:
                params[param_name] = value

        declare_block = "\n".join(declares)
        sql = f"{declare_block}\nUPDATE incidents SET {', '.join(set_clauses)} WHERE id = $id"

        def _upd(session):
            from cloud.db.ydb_client import execute_query

            execute_query(session, sql, params)

        pool.retry_operation_sync(_upd)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_incident(self, incident_id: str) -> Incident | None:
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            from cloud.db.ydb_client import execute_query

            result = execute_query(
                session,
                "DECLARE $id AS Utf8; SELECT * FROM incidents WHERE id = $id",
                {"$id": incident_id},
            )
            rows = result[0].rows
            return self._row_to_incident(rows[0]) if rows else None

        return pool.retry_operation_sync(_q)

    def get_all_incidents(self) -> list[Incident]:
        """Return all incidents ordered by created_at descending."""
        from cloud.db.ydb_client import get_pool

        pool = get_pool()

        def _q(session):
            from cloud.db.ydb_client import execute_query

            result = execute_query(
                session,
                "SELECT * FROM incidents ORDER BY created_at DESC",
            )
            return [self._row_to_incident(row) for row in result[0].rows]

        return pool.retry_operation_sync(_q)

    def get_active_incident_for_chat(self, chat_id: int) -> Incident | None:
        """Look up active incident via in-memory chat mapping, then fetch from YDB."""
        incident_id = _chat_to_incident.get(chat_id)
        if incident_id is None:
            return None
        return self.get_incident(incident_id)

    def get_stale_incidents(
        self,
        pending_max_age: float = 1800,
        accepted_max_age: float = 3600,
    ) -> list[Incident]:
        """Find stale incidents: pending older than pending_max_age,
        accepted older than accepted_max_age. Mirrors in-memory semantics
        in cloud/db/incidents.py:168-181 -- same return type and field semantics.

        SQL pre-filters at YDB level for efficiency; the final age check is
        re-done in Python so semantics are identical to the in-memory backend
        regardless of how the row set was obtained.
        """
        from cloud.db.ydb_client import execute_query, get_pool

        pool = get_pool()
        now = time.time()
        pending_cutoff = now - pending_max_age
        accepted_cutoff = now - accepted_max_age

        def _q(session):
            result = execute_query(
                session,
                """
                DECLARE $pending_cutoff AS Double;
                DECLARE $accepted_cutoff AS Double;
                SELECT * FROM incidents
                WHERE (status = 'pending' AND created_at < $pending_cutoff)
                   OR (status = 'accepted' AND accepted_at IS NOT NULL
                       AND accepted_at < $accepted_cutoff)
                """,
                {
                    "$pending_cutoff": pending_cutoff,
                    "$accepted_cutoff": accepted_cutoff,
                },
            )
            stale: list[Incident] = []
            for row in result[0].rows:
                if row.status == "pending":
                    if now - row.created_at > pending_max_age:
                        stale.append(self._row_to_incident(row))
                elif row.status == "accepted" and row.accepted_at:
                    if now - row.accepted_at > accepted_max_age:
                        stale.append(self._row_to_incident(row))
            return stale

        return pool.retry_operation_sync(_q)

    def get_recent_nearby_incident(
        self,
        lat: float,
        lon: float,
        radius_m: float = 500,
        max_age_s: float = 300,
    ) -> Incident | None:
        """Find a recent pending/accepted incident near (lat, lon).
        Mirrors cloud/db/incidents.py:184-198 -- YDB has no geospatial
        functions, so SQL pre-filters status+age and Python re-checks
        status/age + does the haversine. Re-checking matches in-memory
        semantics exactly.
        """
        from cloud.db.rangers import _haversine
        from cloud.db.ydb_client import execute_query, get_pool

        pool = get_pool()
        now = time.time()
        cutoff = now - max_age_s

        def _q(session):
            result = execute_query(
                session,
                """
                DECLARE $cutoff AS Double;
                SELECT * FROM incidents
                WHERE status IN ('pending', 'accepted')
                  AND created_at > $cutoff
                """,
                {"$cutoff": cutoff},
            )
            for row in result[0].rows:
                if row.status not in ("pending", "accepted"):
                    continue
                if now - row.created_at > max_age_s:
                    continue
                if _haversine(lat, lon, row.lat, row.lon) <= radius_m:
                    return self._row_to_incident(row)
            return None

        return pool.retry_operation_sync(_q)

    # ------------------------------------------------------------------
    # chat mapping (in-memory, transient)
    # ------------------------------------------------------------------

    def assign_chat_to_incident(self, chat_id: int, incident_id: str) -> None:
        _chat_to_incident[chat_id] = incident_id

    def clear_chat_incident(self, chat_id: int) -> None:
        _chat_to_incident.pop(chat_id, None)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_incident(row) -> Incident:
        """Convert a YDB result row to an Incident dataclass.

        FAUN-38a B2: read back drone_photo_b64, drone_comment, ranger_photo_b64,
        alert_message_ids — they are now persisted by update_incident, so the
        read side must surface them too (otherwise write/read asymmetry causes
        silent data loss after restart).
        """
        raw_alert_ids = getattr(row, "alert_message_ids", None) or ""
        if raw_alert_ids:
            try:
                alert_message_ids = {
                    int(k): v for k, v in json.loads(raw_alert_ids).items()
                }
            except (ValueError, json.JSONDecodeError):
                logger.warning(
                    "alert_message_ids JSON decode failed for incident %s: %r",
                    getattr(row, "id", "?"),
                    raw_alert_ids,
                )
                alert_message_ids = {}
        else:
            alert_message_ids = {}

        return Incident(
            id=row.id,
            audio_class=row.audio_class,
            lat=row.lat,
            lon=row.lon,
            confidence=row.confidence,
            gating_level=row.gating_level,
            status=row.status,
            created_at=getattr(row, "created_at", 0.0) or 0.0,
            district=getattr(row, "district", "") or "",
            accepted_by_chat_id=row.accepted_by_chat_id
            if row.accepted_by_chat_id
            else None,
            accepted_by_name=row.accepted_by_name if row.accepted_by_name else None,
            accepted_at=row.accepted_at if row.accepted_at else None,
            arrived_at=getattr(row, "arrived_at", None) or None,
            response_time_min=getattr(row, "response_time_min", None) or None,
            ranger_report_raw=getattr(row, "ranger_report_raw", None) or None,
            ranger_report_legal=getattr(row, "ranger_report_legal", None) or None,
            resolution_details=getattr(row, "resolution_details", "") or "",
            is_demo=getattr(row, "is_demo", False) or False,
            drone_photo_b64=getattr(row, "drone_photo_b64", None) or None,
            drone_comment=getattr(row, "drone_comment", None) or None,
            ranger_photo_b64=getattr(row, "ranger_photo_b64", None) or None,
            alert_message_ids=alert_message_ids,
        )
