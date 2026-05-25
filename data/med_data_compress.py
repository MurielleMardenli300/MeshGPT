import pickle
import requests
import trimesh
import numpy as np
from pathlib import Path
from tqdm import tqdm

TXT_FILE   = "medshapenet/MedShapeNetDataset_liver_only.txt"
OUTPUT_PKL = "data/liver_dataset.pkl"
VAL_RATIO  = 0.15  
DOWNLOAD_DIR = Path("data/stl_cache")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def url_to_name(url: str) -> str:
    """
    '004329_liver.stl'  →  'liver_004329'
    Flips to {category}_{id} so force_category split works correctly.
    """
    stem = Path(url.split("/")[-1]).stem 
    parts = stem.split("_")     
    mesh_id   = parts[0]
    category  = "_".join(parts[1:])
    return f"{category}_{mesh_id}"    


def download_stl(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def load_mesh(path: Path):
    mesh = trimesh.load(str(path), force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None, None
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def build_pkl(txt_file: str, output_pkl: str):
    urls = [line.strip() for line in open(txt_file) if line.strip()]

    vertices_all, faces_all, names_all = [], [], []

    print(f"Downloading and loading {len(urls)} meshes …")
    for url in tqdm(urls):
        filename = Path(url.split("/")[-1])
        dest     = DOWNLOAD_DIR / filename
        name     = url_to_name(url)

        if not download_stl(url, dest):
            continue

        verts, faces = load_mesh(dest)
        if verts is None:
            print(f"  Skipping invalid mesh: {filename}")
            continue

        vertices_all.append(verts)
        faces_all.append(faces)
        names_all.append(name)

    n_total = len(vertices_all)
    n_val   = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    # Shuffle with fixed seed for reproducibility
    rng  = np.random.default_rng(42)
    idxs = rng.permutation(n_total)
    train_idxs = idxs[:n_train]
    val_idxs   = idxs[n_train:]

    data = {
        'vertices_train': [vertices_all[i] for i in train_idxs],
        'faces_train':    [faces_all[i]    for i in train_idxs],
        'name_train':     [names_all[i]    for i in train_idxs],
        'vertices_val':   [vertices_all[i] for i in val_idxs],
        'faces_val':      [faces_all[i]    for i in val_idxs],
        'name_val':       [names_all[i]    for i in val_idxs],
    }

    print(f"\nDataset summary:")
    print(f"  train : {len(data['vertices_train'])} meshes")
    print(f"  val   : {len(data['vertices_val'])} meshes")
    print(f"  names sample: {data['name_train'][:3]}")

    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved → {output_pkl}")


if __name__ == "__main__":
    build_pkl(TXT_FILE, OUTPUT_PKL)