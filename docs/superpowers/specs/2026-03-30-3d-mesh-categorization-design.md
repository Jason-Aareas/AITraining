# Design Spec: 3D Mesh Object Categorization System
**Date:** 2026-03-30
**Status:** Approved

---

## Context

The user has a dataset of 3D architectural/interior scene meshes stored in a custom binary format. Each mesh represents a single object or part (wall, floor, roof, cabinet door, door handle, etc.) and is labeled with a semantic tag. The goal is to train a model that classifies each mesh into one of 1000 categories, then expose it via a WebUI that loads a scene (list of meshes) and displays each mesh with its predicted tag.

**Key constraints:**
- <10K training samples across 1000 classes (~10 samples/class average) — very sparse
- Custom binary file format (structure to be defined by user)
- Vertices are in world-space (useful for scene-relative disambiguation, e.g., exterior vs interior wall)
- Single NVIDIA GPU training

---

## Architecture

### Data Flow

```
Custom binary mesh file
        ↓  parser.py: decode vertices + faces + sharpness
Raw mesh (V×3 vertices, F×3 faces, V×1 sharpness)
        ↓  dataset.py: sample N=1024 points from surface + extract bbox
Point cloud (N×4: XYZ + sharpness)  +  bbox context (6-dim)
        ↓  augment.py: rotation / scale / jitter / dropout
Augmented point cloud + bbox
        ↓  PointNet++ MSG encoder
Geometry feature (1024-dim)  +  bbox MLP (64-dim)
        ↓  Concatenate → MLP classifier
1000-class logits → predicted tag
```

### Why PointNet++ MSG + bbox context

- PointNet++ multi-scale grouping captures both local geometric details (sharp corners, thin features) and global shape structure
- World-space bounding box center + dimensions passed as a separate 6-dim context vector allows the model to use scene position (exterior walls are at scene edges, interior walls are in the middle)
- This combination enables disambiguation of semantically similar but positionally different categories (interior wall vs exterior wall)

---

## Components

### 1. `data_prep/`

| File | Purpose |
|------|---------|
| `parser.py` | Custom binary file decoder. Format defined in `data_prep/format.yaml`. Returns `(vertices: np.ndarray[V,3], faces: np.ndarray[F,3], sharpness: np.ndarray[V,1])` |
| `dataset.py` | PyTorch `Dataset`. Reads `data/labels.csv`, loads mesh via parser, samples N points by barycentric sampling on faces, appends per-point sharpness, extracts bbox context, applies augmentation if training. Returns `(points: Tensor[N,4], bbox: Tensor[6], label: int)` |
| `augment.py` | Augmentation functions: random rotation (SO3), uniform scale jitter, Gaussian point noise, random point dropout. Applied only during training. |
| `split.py` | Reads `data/labels.csv`, performs stratified train/val/test split (70/15/15). Outputs `data/train.csv`, `data/val.csv`, `data/test.csv`. |
| `classes.py` | Builds and saves `data/classes.json` (tag name → int ID mapping) from `labels.csv`. |

**Label CSV format** (`data/labels.csv`):
```
mesh_path,tag
data/raw/scene01/wall_001.bin,interior_wall
data/raw/scene01/wall_002.bin,exterior_wall
...
```

**Binary format config** (`data_prep/format.yaml`) — placeholder, user will fill in:
```yaml
# Byte order: little_endian | big_endian
byte_order: little_endian
# Fields in order: name, dtype, count (per-vertex or per-face or scalar)
header:
  - {name: vertex_count, dtype: int32, scope: scalar}
  - {name: face_count,   dtype: int32, scope: scalar}
vertex_fields:
  - {name: x,         dtype: float32}
  - {name: y,         dtype: float32}
  - {name: z,         dtype: float32}
  - {name: sharpness, dtype: float32}
face_fields:
  - {name: v0, dtype: int32}
  - {name: v1, dtype: int32}
  - {name: v2, dtype: int32}
```

### 2. `model/`

| File | Purpose |
|------|---------|
| `pointnet2.py` | PointNet++ MSG classifier. Input: `(B, N, 4)` points. Architecture: 3 set abstraction layers with multi-scale grouping → global max pool → MLP head that receives concatenated geometry features + bbox context → 1000-class output. |
| `losses.py` | Focal Loss: `FL(p) = -alpha * (1-p)^gamma * log(p)`. Default `alpha=0.25, gamma=2.0`. Handles class imbalance without requiring explicit class weights. |

### 3. `training/`

| File | Purpose |
|------|---------|
| `config.yaml` | All hyperparameters: `num_points=1024, batch_size=32, lr=0.001, epochs=200, focal_gamma=2.0, num_classes=1000` |
| `train.py` | Training loop: AdamW optimizer, cosine annealing LR, focal loss, checkpoint every 10 epochs, best-val-accuracy checkpoint, TensorBoard logging (loss, top-1/top-5 accuracy, LR) |
| `evaluate.py` | Loads checkpoint, runs inference on val or test set, reports: top-1 accuracy, top-5 accuracy, per-class accuracy, confusion matrix (saved as CSV) |

### 4. `webui/`

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app. `POST /predict`: accepts binary mesh file upload, runs inference, returns `{tag, confidence, top5}`. `POST /scene`: accepts `scene.json`, runs inference on all meshes, returns list of `{mesh_path, tag, confidence}`. |
| `static/index.html` | Single-page app: scene file input, mesh list panel (left), Three.js 3D viewer (right) |
| `static/app.js` | Loads scene JSON, calls `/scene` API, populates mesh list with tag labels and confidence, clicking a mesh highlights it in the 3D viewer |

**Scene file format** (`scene.json`):
```json
{
  "meshes": [
    "data/raw/scene01/wall_001.bin",
    "data/raw/scene01/floor_001.bin",
    "data/raw/scene01/roof_001.bin"
  ]
}
```

---

## Project Structure

```
D:\AITraining\
├── data_prep/
│   ├── parser.py
│   ├── dataset.py
│   ├── augment.py
│   ├── split.py
│   ├── classes.py
│   └── format.yaml          # binary format definition (user fills in)
├── model/
│   ├── pointnet2.py
│   └── losses.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── config.yaml
├── webui/
│   ├── server.py
│   └── static/
│       ├── index.html
│       └── app.js
├── data/
│   ├── raw/                 # custom binary mesh files (user provides)
│   ├── labels.csv           # mesh_path, tag (user provides)
│   ├── classes.json         # generated by classes.py
│   ├── train.csv / val.csv / test.csv  # generated by split.py
├── checkpoints/             # saved model checkpoints
├── runs/                    # TensorBoard logs
├── requirements.txt
└── README.md
```

---

## Dependencies (`requirements.txt`)

```
torch>=2.0
torch-geometric
numpy
scipy
fastapi
uvicorn
python-multipart
pyyaml
tensorboard
scikit-learn
trimesh          # mesh surface sampling utilities
```

---

## Verification Plan

1. **Data prep**: Run `split.py` → verify `train.csv` / `val.csv` / `test.csv` exist with correct counts. Run `classes.py` → verify `classes.json` has 1000 entries.
2. **Parser**: Write a unit test that loads one binary file and prints vertex/face count and first 3 vertices. Verify shapes match expected format.
3. **Training**: Run 2-epoch smoke test with a tiny batch to confirm loss decreases and checkpoint is saved.
4. **Evaluation**: Run `evaluate.py` on val set, confirm top-1/top-5 metrics print and confusion matrix CSV is generated.
5. **WebUI**: Start `uvicorn webui.server:app`, open browser at `http://localhost:8000`, load a `scene.json`, verify mesh list populates with predicted tags.

---

## Open Items (to resolve with user)

- [ ] Binary file format structure — user will provide field layout to fill `format.yaml`
- [ ] Exact number of classes in dataset (user said ~1000; `classes.py` will auto-count from labels)
- [ ] Whether scenes have a scene-level metadata file or are just flat lists of mesh paths
