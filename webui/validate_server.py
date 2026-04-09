import json
import os
import re
import struct
from pathlib import Path

import pyodbc
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

app = FastAPI()


def get_conn():
    return pyodbc.connect(DB_CONN)


# ── Binary .js mesh parser ────────────────────────────────────────────────────

def _scan_section(data: bytes, label: bytes, search_from: int) -> int:
    """Return byte offset immediately after the label bytes, starting at search_from."""
    idx = data.find(label, search_from)
    if idx == -1:
        raise ValueError(f"Section {label!r} not found from offset {search_from}")
    return idx + len(label)


def parse_mesh(data: bytes) -> dict:
    file_len = len(data)

    # Detect version from end-of-file label
    # AareasExportV2: 14 chars at [end-15..end-1]
    # AareasExport:   12 chars at [end-13..end-1]
    if file_len >= 15 and data[file_len - 15:file_len - 1] == b"AareasExportV2":
        version = 2
    elif file_len >= 13 and data[file_len - 13:file_len - 1] == b"AareasExport\x00\x00":
        version = 1
    else:
        # Try without strict null check
        tail = data[max(0, file_len - 20):]
        if b"AareasExportV2" in tail:
            version = 2
        elif b"AareasExport" in tail:
            version = 1
        else:
            raise ValueError("Not a valid Aareas mesh file (no version label)")

    # ── Header (offset 0) ────────────────────────────────────────────────────
    # int32 @ 0: verticesCount
    # int32 @ 4: normalsCount
    # int32 @ 12: uvsChannels
    uvs_channels = struct.unpack_from("<i", data, 12)[0]

    # ── Transform matrix (offset 31, 16×float32 = 64 bytes) ─────────────────
    matrix = list(struct.unpack_from("<16f", data, 31))

    # Scale: V2 uses matrix[0] as the unit scale; V1 uses 0.0254 (inches→metres)
    transform = matrix[0] if version == 2 else 0.0254

    # Translation components from matrix (column-major 4×4):
    #   matrix[12]=tx, matrix[13]=ty, matrix[14]=tz  (3ds Max Z-up space)
    tx = matrix[12]
    ty = matrix[13]
    tz = matrix[14]

    # ── Geometry section (offset 95) ─────────────────────────────────────────
    # Layout inside geometry view:
    #   "vertices\0" (9 bytes) | vertexCount(int32) | V×3 float32
    #   "normals\0"  (8 bytes) | normalCount(int32) | N×3 float32
    #   "colors\0"   (7 bytes) | colorCount(int32)  | C×3 float32
    #   [for each UV channel]  | label(3 bytes) | count(int32) | data
    #   "faces\0"    (6 bytes) | faceCount(int32)   | F×stride int32

    GEO_START = 95

    # vertices section
    pos = _scan_section(data, b"vertices\x00", GEO_START)
    v_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    raw_verts = struct.unpack_from(f"<{v_count * 3}f", data, pos)
    pos += v_count * 3 * 4

    # Apply coordinate transform matching Scene.tsx parseAareasBinary + Rx(-90°) group rotation:
    #   Step 1 (Scene.tsx):  vx=(m12+rx)*s,  vy=(-m14+ry)*s,  vz=(m13+rz)*s
    #   Step 2 (Rx -90°):    X=vx,  Y=vz,  Z=-vy
    #   Combined:
    #     three.X = (m12 + rx) * s
    #     three.Y = (m13 + rz) * s   ← matrix[13] + Z-vertex (height)
    #     three.Z = (m14 - ry) * s   ← matrix[14] - Y-vertex (depth, negated)
    m12, m13, m14 = matrix[12], matrix[13], matrix[14]
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

    # normals section
    pos = _scan_section(data, b"normals\x00", pos)
    n_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    raw_norms = struct.unpack_from(f"<{n_count * 3}f", data, pos)
    pos += n_count * 3 * 4
    normals = [list(raw_norms[i * 3:i * 3 + 3]) for i in range(n_count)]

    # colors section
    pos = _scan_section(data, b"colors\x00", pos)
    c_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4 + c_count * 3 * 4  # skip color data

    # Count actual UV sections and skip their data.
    # Format: "uvs\x00" (4 bytes) + channel_idx (1 byte) + count (int32) + count×2×float32
    # stride = 8 + 3 × actual_uv_sections  (0→8, 1→11, 2→14)
    actual_uv_count = 0
    while True:
        uvs_idx = data.find(b"uvs\x00", pos)
        faces_idx = data.find(b"faces\x00", pos)
        if uvs_idx == -1 or (faces_idx != -1 and faces_idx <= uvs_idx):
            break
        uv_data_pos = uvs_idx + 4 + 1  # skip label + channel_idx byte
        uv_count = struct.unpack_from("<i", data, uv_data_pos)[0]
        pos = uv_data_pos + 4 + uv_count * 2 * 4
        actual_uv_count += 1

    pos = _scan_section(data, b"faces\x00", pos)
    faces_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4

    # stride: type(1) + v0,v1,v2(3) + uv_indices(3×actual_uv) + fn(1) + vn0,vn1,vn2(3)
    stride = 8 + 3 * actual_uv_count
    total_ints = faces_count * stride

    if pos + total_ints * 4 > file_len:
        raise ValueError(
            f"Face data overruns file: need {total_ints * 4} bytes at {pos}, "
            f"file is {file_len} bytes"
        )

    raw_faces = struct.unpack_from(f"<{total_ints}i", data, pos)

    # Vertex indices are always at positions [base+1], [base+2], [base+3]
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


# ── API routes ────────────────────────────────────────────────────────────────

def _is_valid_scene_name(name: str) -> bool:
    """Keep only scenes whose name starts with a 4-digit number in [4000, 8000]."""
    m = re.match(r'^(\d{4})', name or '')
    if not m:
        return False
    n = int(m.group(1))
    return 4000 <= n <= 8000


@app.get("/api/scenes")
def list_scenes():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT Id, SceneName FROM Scenes WHERE IsLive=1 ORDER BY SceneName")
    rows = [{"id": r[0], "name": r[1]} for r in cursor.fetchall()
            if _is_valid_scene_name(r[1])]
    conn.close()
    return rows


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

    result = []
    for obj in objects:
        mesh_file = obj.get("Mesh", "")
        key = mesh_file.strip().lower()
        info = tag_map.get(key, {})
        full_path = mesh_dir / mesh_file
        result.append(
            {
                "mesh_file": mesh_file,
                "surface_name": info.get("surface_name") or "",
                "tag": info.get("tag") or "",
                "has_tag": bool(info.get("tag")),
                "file_exists": full_path.exists(),
            }
        )

    tagged = sum(1 for m in result if m["has_tag"])
    return {"scene_id": scene_id, "total": len(result), "tagged": tagged, "meshes": result}


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


# ── Static files ──────────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(static_dir / "validate.html"))
