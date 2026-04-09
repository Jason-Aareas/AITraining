"""
Import training data from MS SQL → local SQLite (data/training.db).

Usage:
    python data_prep/import_db.py            # full import (all 310 valid scenes)
    python data_prep/import_db.py --scene 70176   # add/refresh one scene
    python data_prep/import_db.py --list     # list what is already imported
    python data_prep/import_db.py --stats    # summary statistics
"""

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

import pyodbc
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

LOCAL_DB_PATH = ROOT / "data" / "training.db"


# ── Connection helpers ────────────────────────────────────────────────────────

def _build_mssql_conn() -> str:
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


def get_local_db() -> sqlite3.Connection:
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LOCAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_schema(ldb: sqlite3.Connection):
    ldb.executescript("""
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
    ldb.commit()


# ── Validation ────────────────────────────────────────────────────────────────

def _is_valid_scene_name(name: str) -> bool:
    """Keep only scenes with 4-digit prefix in [4000, 8000]."""
    m = re.match(r'^(\d{4})', name or '')
    if not m:
        return False
    return 4000 <= int(m.group(1)) <= 8000


# ── Import logic ──────────────────────────────────────────────────────────────

def import_scene(cur, ldb: sqlite3.Connection, scene_id: int, scene_name: str, staging_url: str) -> int:
    """
    Import one scene from MS SQL into SQLite.
    Returns the number of mesh records written.
    """
    # Upsert scene record
    ldb.execute("""
        INSERT INTO scenes (id, scene_name, staging_url, imported_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            scene_name  = excluded.scene_name,
            staging_url = excluded.staging_url,
            imported_at = excluded.imported_at
    """, (scene_id, scene_name, staging_url))

    # Fetch all tagged mesh surfaces for this scene
    cur.execute("""
        SELECT ss.MeshPath, ss.SurfaceName, surf.SurfaceName AS tag
        FROM   SceneSurfaces ss
        LEFT JOIN Surfaces surf ON surf.Id = ss.MasterSurfaceId
        WHERE  ss.SceneId = ? AND ss.Deleted = 0 AND ss.MeshPath IS NOT NULL
    """, (scene_id,))
    rows = cur.fetchall()

    # Replace all mesh records for this scene (clean re-import)
    ldb.execute("DELETE FROM meshes WHERE scene_id = ?", (scene_id,))
    ldb.executemany("""
        INSERT OR REPLACE INTO meshes (scene_id, mesh_file, original_tag, surface_name, imported_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, [
        (scene_id, r[0].strip(), r[2], r[1])
        for r in rows if r[0]
    ])

    return len(rows)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_import_all(cur, ldb: sqlite3.Connection):
    cur.execute("SELECT Id, SceneName, StagingURL FROM Scenes WHERE IsLive=1 ORDER BY SceneName")
    valid = [(r[0], r[1], r[2]) for r in cur.fetchall() if _is_valid_scene_name(r[1])]

    print(f"Importing {len(valid)} valid scenes into SQLite...")
    total_meshes = 0
    for i, (sid, sname, url) in enumerate(valid, 1):
        count = import_scene(cur, ldb, sid, sname, url)
        total_meshes += count
        print(f"  [{i:3d}/{len(valid)}] {sname}: {count} meshes")
        if i % 20 == 0:
            ldb.commit()  # commit in batches to avoid huge transactions

    ldb.commit()
    print(f"\nDone. {len(valid)} scenes, {total_meshes} total mesh records.")


def cmd_import_one(cur, ldb: sqlite3.Connection, scene_id: int):
    cur.execute("SELECT Id, SceneName, StagingURL FROM Scenes WHERE Id=?", (scene_id,))
    row = cur.fetchone()
    if not row:
        print(f"ERROR: Scene {scene_id} not found in MS SQL.", file=sys.stderr)
        sys.exit(1)
    sid, sname, url = row[0], row[1], row[2]
    if not _is_valid_scene_name(sname):
        print(f"ERROR: Scene '{sname}' is a test scene (outside [4000,8000]).", file=sys.stderr)
        sys.exit(1)
    count = import_scene(cur, ldb, sid, sname, url)
    ldb.commit()
    print(f"Imported scene '{sname}' (id={sid}): {count} mesh records.")


def cmd_list(ldb: sqlite3.Connection):
    rows = ldb.execute("""
        SELECT s.id, s.scene_name,
               COUNT(m.mesh_file)                                              AS total,
               SUM(CASE WHEN m.original_tag IS NOT NULL THEN 1 ELSE 0 END)    AS tagged,
               s.imported_at
        FROM   scenes s
        LEFT JOIN meshes m ON m.scene_id = s.id
        GROUP BY s.id
        ORDER BY s.scene_name
    """).fetchall()
    print(f"{'ID':>7}  {'Scene':<52}  {'Meshes':>7}  {'Tagged':>7}  Imported")
    print("-" * 100)
    for r in rows:
        print(f"{r['id']:>7}  {r['scene_name']:<52}  {r['total']:>7}  {r['tagged']:>7}  {r['imported_at']}")
    print(f"\nTotal: {len(rows)} scenes imported.")


def cmd_stats(ldb: sqlite3.Connection):
    total_scenes  = ldb.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    total_meshes  = ldb.execute("SELECT COUNT(*) FROM meshes").fetchone()[0]
    tagged_meshes = ldb.execute("SELECT COUNT(*) FROM meshes WHERE original_tag IS NOT NULL").fetchone()[0]
    corrections   = ldb.execute("SELECT COUNT(*) FROM tag_corrections").fetchone()[0]
    excluded      = ldb.execute("SELECT COUNT(*) FROM tag_corrections WHERE corrected_tag IS NULL").fetchone()[0]
    tag_count     = ldb.execute("SELECT COUNT(DISTINCT original_tag) FROM meshes WHERE original_tag IS NOT NULL").fetchone()[0]

    # Effective training samples (production tag OR corrected tag, minus excluded)
    effective = ldb.execute("""
        SELECT COUNT(*) FROM meshes m
        LEFT JOIN tag_corrections c
               ON c.scene_id = m.scene_id AND LOWER(c.mesh_file) = LOWER(m.mesh_file)
        WHERE (c.scene_id IS NOT NULL AND c.corrected_tag IS NOT NULL)
           OR (c.scene_id IS NULL     AND m.original_tag  IS NOT NULL)
    """).fetchone()[0]

    print(f"Scenes imported   : {total_scenes}")
    print(f"Mesh records      : {total_meshes}")
    print(f"  Tagged (DB)     : {tagged_meshes}")
    print(f"  Distinct tags   : {tag_count}")
    print(f"Corrections       : {corrections}  ({excluded} excluded)")
    print(f"Effective training: {effective} samples")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync training data: MS SQL → SQLite")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--scene", type=int, metavar="ID",
                     help="Import / refresh a single scene by MS SQL ID")
    grp.add_argument("--list",  action="store_true", help="List already-imported scenes")
    grp.add_argument("--stats", action="store_true", help="Show summary statistics")
    args = parser.parse_args()

    ldb = get_local_db()
    init_schema(ldb)

    if args.list:
        cmd_list(ldb)
        ldb.close()
        sys.exit(0)

    if args.stats:
        cmd_stats(ldb)
        ldb.close()
        sys.exit(0)

    # All other commands need MS SQL
    print("Connecting to MS SQL...")
    mssql = pyodbc.connect(_build_mssql_conn())
    cur   = mssql.cursor()

    if args.scene:
        cmd_import_one(cur, ldb, args.scene)
    else:
        cmd_import_all(cur, ldb)

    mssql.close()
    ldb.close()
