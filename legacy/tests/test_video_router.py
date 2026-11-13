"""Tests del router de video (helpers puros, sin lanzar el pipeline)."""
from __future__ import annotations

from src.api.routers.video import _buscar_video_por_checksum
from src.database.models import Video


def _video(db, checksum: str, status: str | None = "active") -> Video:
    v = Video(
        original_filename = "v.mp4",
        storage_key       = "data/videos/v.mp4",
        checksum          = checksum,
        retention_status  = status,
    )
    db.add(v)
    db.commit()
    return v


def test_buscar_video_por_checksum(db_session):
    v = _video(db_session, "abc123")
    assert _buscar_video_por_checksum(db_session, "abc123").id == v.id
    assert _buscar_video_por_checksum(db_session, "otro") is None


def test_buscar_video_ignora_borrados(db_session):
    _video(db_session, "abc123", status="deleted")
    assert _buscar_video_por_checksum(db_session, "abc123") is None


def test_buscar_video_devuelve_el_mas_reciente(db_session):
    _video(db_session, "dup")
    nuevo = _video(db_session, "dup")
    assert _buscar_video_por_checksum(db_session, "dup").id == nuevo.id
