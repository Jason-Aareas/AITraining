import json
import os
import re
import sqlite3
import struct
from pathlib import Path
from typing import Optional

import pyodbc
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")


def _build_pyodbc_conn() -> str:
    meshdb = os.getenv("MESHDB", "")
    parts = {}
    for item in meshdb.strip('"').split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            parts[k.strip().lower()] = v.strip()
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"Server={parts.get('server', '')};"
        f"Database={parts.get('database', '')};"
        f"UID={parts.get('uid', '')};"
        f"PWD={parts.get('pwd', '')};"
        "TrustServerCertificate=yes"
    )


DB_CONN = _build_pyodbc_conn()
STAGING_ROOT = os.getenv("STAGING_ROOT", "//192.168.100.44/Library/STAGING")

# Local SQLite for tag corrections
LOCAL_DB_PATH = Path(__file__).parent.parent / "data" / "training.db"
LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _cleared_name(mesh_file: str) -> str:
    """SQLite UDF: strip trailing dimension/version suffixes + extension.
    'Crown13.js'          → 'Crown'
    'COUNTERTOP_0009_.js' → 'COUNTERTOP'
    'CAB_DOOR_0001_17x25.js' → 'CAB_DOOR'
    """
    if not mesh_file:
        return ""
    stem = Path(mesh_file).stem
    stem = re.sub(r'^\d+_', '', stem)   # strip leading number prefix: '1_CAB_BASE' -> 'CAB_BASE'
    prev = None
    while stem != prev:
        prev = stem
        stem = re.sub(r'_?\d+([xX]\d+)+$', '', stem)  # dimension suffix: _17x25 / 10X20 / 30x30x30
        stem = re.sub(r'[_\d]+$', '', stem)          # trailing digits / underscores
    return stem


def get_local_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(LOCAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.create_function("cleared_name", 1, _cleared_name)
    return conn


def _init_local_db():
    conn = get_local_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scenes (
            id          INTEGER PRIMARY KEY,
            scene_name  TEXT NOT NULL,
            staging_url TEXT NOT NULL,
            imported_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS meshes (
            scene_id     INTEGER NOT NULL,
            mesh_file    TEXT    NOT NULL,
            original_tag TEXT,
            surface_name TEXT,
            imported_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (scene_id, mesh_file),
            FOREIGN KEY (scene_id) REFERENCES scenes(id)
        );

        CREATE TABLE IF NOT EXISTS tag_corrections (
            scene_id      INTEGER NOT NULL,
            mesh_file     TEXT    NOT NULL,
            corrected_tag TEXT,
            note          TEXT    DEFAULT '',
            updated_at    TEXT    DEFAULT (datetime('now')),
            PRIMARY KEY (scene_id, mesh_file)
        );

        CREATE INDEX IF NOT EXISTS idx_meshes_scene ON meshes(scene_id);
        CREATE INDEX IF NOT EXISTS idx_meshes_tag   ON meshes(original_tag);
    """)
    conn.commit()
    conn.close()


_init_local_db()

app = FastAPI()


def get_conn():
    return pyodbc.connect(DB_CONN)


# ── Binary .js mesh parser ────────────────────────────────────────────────────

def _scan_section(data: bytes, label: bytes, search_from: int) -> int:
    idx = data.find(label, search_from)
    if idx == -1:
        raise ValueError(f"Section {label!r} not found from offset {search_from}")
    return idx + len(label)


def parse_mesh(data: bytes) -> dict:
    file_len = len(data)

    if file_len >= 15 and data[file_len - 15:file_len - 1] == b"AareasExportV2":
        version = 2
    elif file_len >= 13 and data[file_len - 13:file_len - 1] == b"AareasExport\x00\x00":
        version = 1
    else:
        tail = data[max(0, file_len - 20):]
        if b"AareasExportV2" in tail:
            version = 2
        elif b"AareasExport" in tail:
            version = 1
        else:
            raise ValueError("Not a valid Aareas mesh file (no version label)")

    uvs_channels = struct.unpack_from("<i", data, 12)[0]
    matrix = list(struct.unpack_from("<16f", data, 31))
    transform = matrix[0] if version == 2 else 0.0254
    m12, m13, m14 = matrix[12], matrix[13], matrix[14]

    GEO_START = 95

    pos = _scan_section(data, b"vertices\x00", GEO_START)
    v_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    raw_verts = struct.unpack_from(f"<{v_count * 3}f", data, pos)
    pos += v_count * 3 * 4

    # Coordinate transform: Scene.tsx parseAareasBinary + Rx(-90°) group rotation
    vertices = []
    for i in range(v_count):
        rx = raw_verts[i * 3]
        ry = raw_verts[i * 3 + 1]
        rz = raw_verts[i * 3 + 2]
        vertices.append([
            (m12 + rx) * transform,
            (m13 + rz) * transform,
            (m14 - ry) * transform,
        ])

    pos = _scan_section(data, b"normals\x00", pos)
    n_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    raw_norms = struct.unpack_from(f"<{n_count * 3}f", data, pos)
    pos += n_count * 3 * 4
    normals = [list(raw_norms[i * 3:i * 3 + 3]) for i in range(n_count)]

    pos = _scan_section(data, b"colors\x00", pos)
    c_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4 + c_count * 3 * 4

    actual_uv_count = 0
    while True:
        uvs_idx = data.find(b"uvs\x00", pos)
        faces_idx = data.find(b"faces\x00", pos)
        if uvs_idx == -1 or (faces_idx != -1 and faces_idx <= uvs_idx):
            break
        uv_data_pos = uvs_idx + 4 + 1
        uv_count = struct.unpack_from("<i", data, uv_data_pos)[0]
        pos = uv_data_pos + 4 + uv_count * 2 * 4
        actual_uv_count += 1

    pos = _scan_section(data, b"faces\x00", pos)
    faces_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    stride = 8 + 3 * actual_uv_count
    total_ints = faces_count * stride

    if pos + total_ints * 4 > file_len:
        raise ValueError(
            f"Face data overruns file: need {total_ints * 4} bytes at {pos}, "
            f"file is {file_len} bytes"
        )

    raw_faces = struct.unpack_from(f"<{total_ints}i", data, pos)
    faces = []
    for i in range(faces_count):
        base = i * stride
        faces.append([raw_faces[base + 1], raw_faces[base + 2], raw_faces[base + 3]])

    return {
        "vertex_count": v_count,
        "tri_count": len(faces),
        "vertices": vertices,
        "normals": normals,
        "faces": faces,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_valid_scene_name(name: str) -> bool:
    m = re.match(r'^(\d{4})', name or '')
    if not m:
        return False
    n = int(m.group(1))
    return 4000 <= n <= 8000


def _load_corrections(scene_id: int) -> dict:
    """Return {mesh_file_lower: {corrected_tag, note}} from local DB."""
    ldb = get_local_db()
    rows = ldb.execute(
        "SELECT mesh_file, corrected_tag, note FROM tag_corrections WHERE scene_id=?",
        (scene_id,)
    ).fetchall()
    ldb.close()
    return {r["mesh_file"].strip().lower(): dict(r) for r in rows}


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/scenes")
def list_scenes():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT Id, SceneName FROM Scenes WHERE IsLive=1 ORDER BY SceneName")
    rows = [{"id": r[0], "name": r[1]} for r in cursor.fetchall()
            if _is_valid_scene_name(r[1])]
    conn.close()
    return rows


@app.get("/api/tags")
def list_tags():
    """Return all unique tag names from production DB, sorted."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT SurfaceName FROM Surfaces WHERE SurfaceName IS NOT NULL ORDER BY SurfaceName"
    )
    tags = [r[0] for r in cursor.fetchall() if r[0]]
    conn.close()
    return tags


@app.get("/api/scene/{scene_id}")
def get_scene(scene_id: int):
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT StagingURL FROM Scenes WHERE Id=?", scene_id)
    row = cursor.fetchone()
    if not row or not row[0]:
        raise HTTPException(404, "Scene not found or no StagingURL")
    staging_url = row[0]

    jslist_path = Path(STAGING_ROOT) / staging_url
    if not jslist_path.exists():
        raise HTTPException(404, f"JSList not found: {jslist_path}")

    with open(jslist_path) as f:
        jslist = json.load(f)

    mesh_dir = jslist_path.parent
    objects = jslist.get("Objects", [])

    cursor.execute(
        """
        SELECT ss.MeshPath, ss.SurfaceName, surf.SurfaceName
        FROM SceneSurfaces ss
        LEFT JOIN Surfaces surf ON surf.Id = ss.MasterSurfaceId
        WHERE ss.SceneId = ? AND ss.Deleted = 0
        """,
        scene_id,
    )
    tag_map = {}
    for mesh_path, surface_name, tag in cursor.fetchall():
        if mesh_path:
            key = mesh_path.strip().lower()
            tag_map[key] = {"surface_name": surface_name, "tag": tag}

    conn.close()

    # Load local corrections
    corrections = _load_corrections(scene_id)

    result = []
    for obj in objects:
        mesh_file = obj.get("Mesh", "")
        key = mesh_file.strip().lower()
        info = tag_map.get(key, {})
        original_tag = info.get("tag") or ""
        full_path = mesh_dir / mesh_file

        # Merge with local correction
        if key in corrections:
            c = corrections[key]
            corrected_tag = c["corrected_tag"]  # None = excluded
            excluded = corrected_tag is None
            effective_tag = "" if excluded else corrected_tag
            corrected = True
        else:
            corrected = False
            excluded = False
            effective_tag = original_tag

        result.append({
            "mesh_file": mesh_file,
            "surface_name": info.get("surface_name") or "",
            "tag": effective_tag,
            "original_tag": original_tag,
            "has_tag": bool(effective_tag),
            "corrected": corrected,
            "excluded": excluded,
            "file_exists": full_path.exists(),
        })

    tagged = sum(1 for m in result if m["has_tag"])
    corrected_count = sum(1 for m in result if m["corrected"])
    excluded_count = sum(1 for m in result if m["excluded"])
    return {
        "scene_id": scene_id,
        "total": len(result),
        "tagged": tagged,
        "corrected": corrected_count,
        "excluded": excluded_count,
        "meshes": result,
    }


@app.get("/api/mesh/{scene_id}/{mesh_file:path}")
def get_mesh(scene_id: int, mesh_file: str):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT StagingURL FROM Scenes WHERE Id=?", scene_id)
    row = cursor.fetchone()
    conn.close()
    if not row or not row[0]:
        raise HTTPException(404, "Scene not found")

    jslist_path = Path(STAGING_ROOT) / row[0]
    mesh_path = jslist_path.parent / mesh_file
    if not mesh_path.exists():
        raise HTTPException(404, f"Mesh file not found: {mesh_path}")

    with open(mesh_path, "rb") as f:
        raw = f.read()

    try:
        return parse_mesh(raw)
    except Exception as e:
        raise HTTPException(500, f"Failed to parse mesh: {e}")


# ── Tag correction endpoints ──────────────────────────────────────────────────

class CorrectionRequest(BaseModel):
    corrected_tag: Optional[str] = None   # None = exclude from training
    note: Optional[str] = ""


@app.post("/api/correction/{scene_id}/{mesh_file:path}")
def set_correction(scene_id: int, mesh_file: str, body: CorrectionRequest):
    """Upsert a tag correction (or exclusion) for a specific mesh."""
    ldb = get_local_db()
    ldb.execute(
        """
        INSERT INTO tag_corrections (scene_id, mesh_file, corrected_tag, note, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(scene_id, mesh_file) DO UPDATE SET
            corrected_tag = excluded.corrected_tag,
            note          = excluded.note,
            updated_at    = excluded.updated_at
        """,
        (scene_id, mesh_file.strip(), body.corrected_tag, body.note or "")
    )
    ldb.commit()
    ldb.close()
    return {"ok": True}


@app.delete("/api/correction/{scene_id}/{mesh_file:path}")
def delete_correction(scene_id: int, mesh_file: str):
    """Remove a local correction — reverts to production DB tag."""
    ldb = get_local_db()
    ldb.execute(
        "DELETE FROM tag_corrections WHERE scene_id=? AND mesh_file=?",
        (scene_id, mesh_file.strip())
    )
    ldb.commit()
    ldb.close()
    return {"ok": True}


@app.get("/api/corrections/export")
def export_corrections():
    """Export all corrections as JSON (for review / backup)."""
    ldb = get_local_db()
    rows = ldb.execute(
        "SELECT scene_id, mesh_file, corrected_tag, note, updated_at FROM tag_corrections ORDER BY scene_id, mesh_file"
    ).fetchall()
    ldb.close()
    return [dict(r) for r in rows]


# ── Batch tag fix endpoints ───────────────────────────────────────────────────

@app.get("/api/batch-groups")
def batch_groups(q: str = ""):
    """
    Return DISTINCT (cleared_name, effective_tag) groups across all scenes.
    effective_tag = corrected_tag if correction exists, else original_tag.
    Optional ?q= filters by cleared_name (case-insensitive substring).
    """
    ldb = get_local_db()
    search = q.strip().lower()
    rows = ldb.execute(
        """
        SELECT cleared_name(m.mesh_file)                       AS cname,
               COALESCE(c.corrected_tag, m.original_tag)       AS eff_tag,
               COUNT(*)                                         AS mesh_count,
               COUNT(DISTINCT m.scene_id)                      AS scene_count
        FROM   meshes m
        LEFT JOIN tag_corrections c
               ON c.scene_id = m.scene_id
              AND LOWER(c.mesh_file) = LOWER(m.mesh_file)
        WHERE  (? = '' OR LOWER(cleared_name(m.mesh_file)) LIKE '%'||?||'%')
        GROUP BY cname, eff_tag
        ORDER BY cname, eff_tag NULLS LAST
        """,
        (search, search),
    ).fetchall()
    ldb.close()

    # Exclude: cleared name is purely digits+"v" (e.g. "001v", "1v") with no tag
    _version_re = re.compile(r'^\d+[vV]$')
    rows = [r for r in rows if not (_version_re.match(r['cname'] or '') and not r['eff_tag'])]

    # Exclude: cleared name contains "spline" — handled separately, not a mesh category
    rows = [r for r in rows if 'spline' not in (r['cname'] or '').lower()]

    return rows


class BatchFixRequest(BaseModel):
    cleared_name: str
    new_tag: Optional[str] = None   # None = exclude from training


@app.post("/api/batch-fix")
def batch_fix(body: BatchFixRequest):
    """
    Set tag for ALL meshes whose cleared_name matches body.cleared_name.
    - new_tag == original_tag  → delete correction (revert to DB)
    - new_tag != original_tag  → upsert correction
    - new_tag is None          → upsert exclusion
    """
    ldb = get_local_db()
    rows = ldb.execute(
        "SELECT scene_id, mesh_file, original_tag FROM meshes WHERE cleared_name(mesh_file) = ?",
        (body.cleared_name,),
    ).fetchall()

    updated = 0
    for row in rows:
        sid, mf, orig = row["scene_id"], row["mesh_file"], row["original_tag"]
        if body.new_tag is not None and body.new_tag == orig:
            # Revert: remove any existing correction
            ldb.execute(
                "DELETE FROM tag_corrections WHERE scene_id=? AND mesh_file=?",
                (sid, mf),
            )
        else:
            ldb.execute(
                """
                INSERT INTO tag_corrections (scene_id, mesh_file, corrected_tag, note, updated_at)
                VALUES (?, ?, ?, 'batch fix', datetime('now'))
                ON CONFLICT(scene_id, mesh_file) DO UPDATE SET
                    corrected_tag = excluded.corrected_tag,
                    note          = excluded.note,
                    updated_at    = excluded.updated_at
                """,
                (sid, mf, body.new_tag),
            )
        updated += 1

    ldb.commit()
    ldb.close()
    return {"ok": True, "updated": updated}


# ── Static files ──────────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(static_dir / "validate.html"))


@app.get("/batch")
def batch_page():
    return FileResponse(str(static_dir / "batch.html"))
