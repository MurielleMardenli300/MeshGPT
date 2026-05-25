import random
import sys
from model.pointnet import get_pointnet_classifier
import omegaconf
import torch
from pathlib import Path

import trimesh
import numpy as np
from scipy import linalg

from dataset.quantized_soup import QuantizedSoupTripletsCreator
from dataset.triangles import TriangleNodesWithFacesAndSequenceIndices
from trainer import get_rvqvae_v0_decoder
from trainer.train_transformer import get_qsoup_model_config
from util.meshlab import meshlab_proc
from util.misc import get_parameters_from_state_dict
from util.visualization import plot_vertices_and_faces
from tqdm import tqdm
from model.transformer import QuantSoupTransformer
from pytorch_lightning import seed_everything


# ─────────────────────────────────────────────
#  POINT CLOUD HELPERS
# ─────────────────────────────────────────────

MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".off"}

def normalize_pointcloud(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.astype(np.float32)


def sample_pointcloud(vertices: np.ndarray, faces: np.ndarray, n_points: int = 2048) -> np.ndarray:
    """Sample a point cloud from vertices + faces directly (no file I/O needed)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return normalize_pointcloud(pts)

def load_all_meshes(mesh_dir: Path, n_points: int = 2048) -> np.ndarray:
    """Load all meshes from a directory of .obj/.stl files → stacked point clouds (N, n_points, 3)."""
    paths = [p for p in sorted(mesh_dir.rglob("*")) if p.suffix.lower() in MESH_EXTENSIONS]
    if not paths:
        raise ValueError(f"No mesh files found in {mesh_dir}")
    pcs = []
    for p in tqdm(paths, desc=f"Loading {mesh_dir.name}"):
        try:
            mesh = trimesh.load(str(p), force='mesh')
            pts, _ = trimesh.sample.sample_surface(mesh, n_points)
            pcs.append(normalize_pointcloud(pts))
        except Exception as e:
            print(f"  Skipping {p.name}: {e}")
    return np.stack(pcs)

def load_reference_pcs_from_dataset(config, n_points: int = 2048) -> np.ndarray:
    """
    Load reference point clouds directly from the pkl dataset used for training,
    using the same TriangleNodes class — no mesh files needed.
    Uses the 'val' split by default so it matches evaluation conditions.
    """
    dataset = TriangleNodesWithFacesAndSequenceIndices(config, split='val',
                            scale_augment=False, shift_augment=False,
                            force_category=config.ft_category)
    pcs = []
    for verts, faces in tqdm(
        zip(dataset.cached_vertices, dataset.cached_faces),
        total=len(dataset.cached_vertices),
        desc="Loading reference point clouds from pkl"
    ):
        try:
            pcs.append(sample_pointcloud(np.array(verts), np.array(faces), n_points))
        except Exception as e:
            print(f"  Skipping a reference mesh: {e}")

    if not pcs:
        raise ValueError("No valid reference meshes could be loaded from the dataset pkl.")
    return np.stack(pcs)  # (N, n_points, 3)


def pointclouds_to_features(gen_pcs: np.ndarray, ref_pcs: np.ndarray,
                             n_components: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on the reference set, then project both sets into that shared space.
    This ensures gen and ref features are always comparable regardless of set size.
    
    n_components is capped at min(n_ref_shapes, n_points*3) to avoid rank issues.
    """
    N_ref = ref_pcs.shape[0]
    ref_flat = ref_pcs.reshape(N_ref, -1)           # (N_ref, n_points*3)
    gen_flat = gen_pcs.reshape(len(gen_pcs), -1)    # (N_gen, n_points*3)

    # Fit mean/std on reference only
    ref_mean = ref_flat.mean(0)
    ref_std  = ref_flat.std(0) + 1e-8
    ref_norm = (ref_flat - ref_mean) / ref_std
    gen_norm = (gen_flat - ref_mean) / ref_std      # same normalization

    # PCA on reference only — cap components to avoid SVD rank issues
    n_components = min(n_components, N_ref, ref_norm.shape[1])
    _, _, Vt = np.linalg.svd(ref_norm, full_matrices=False)
    V = Vt[:n_components].T                         # (n_points*3, n_components)

    ref_feats = ref_norm @ V                        # (N_ref, n_components)
    gen_feats = gen_norm @ V                        # (N_gen, n_components)

    return gen_feats, ref_feats


# ─────────────────────────────────────────────
#  FID
# ─────────────────────────────────────────────

def compute_fid(feats_gen: np.ndarray, feats_real: np.ndarray) -> float:
    mu_g, sig_g = feats_gen.mean(0),  np.cov(feats_gen,  rowvar=False)
    mu_r, sig_r = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    diff = mu_g - mu_r
    covmean, _ = linalg.sqrtm(sig_g @ sig_r, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_g + sig_r - 2 * covmean))


# ─────────────────────────────────────────────
#  KID
# ─────────────────────────────────────────────

def compute_kid(feats_gen: np.ndarray, feats_real: np.ndarray,
                subset_size: int = 100, n_subsets: int = 100) -> tuple[float, float]:
    m = min(len(feats_gen), len(feats_real), subset_size)
    scores = []
    for _ in range(n_subsets):
        r = feats_real[np.random.choice(len(feats_real), m, replace=False)]
        g = feats_gen [np.random.choice(len(feats_gen),  m, replace=False)]
        d = r.shape[1]
        k_rr = (r @ r.T / d + 1) ** 3
        k_gg = (g @ g.T / d + 1) ** 3
        k_rg = (r @ g.T / d + 1) ** 3
        scores.append(k_rr.mean() + k_gg.mean() - 2 * k_rg.mean())
    return float(np.mean(scores)), float(np.std(scores))


# ─────────────────────────────────────────────
#  MMD  (Minimum Matching Distance)
# ─────────────────────────────────────────────

def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    diff  = pc1[:, None, :] - pc2[None, :, :]       # (N, M, 3)
    dists = (diff ** 2).sum(-1)                      # (N, M)
    return float(dists.min(1).mean() + dists.min(0).mean())


def compute_mmd(gen_pcs: np.ndarray, ref_pcs: np.ndarray) -> float:
    """For each generated shape, find the closest reference shape; average those distances."""
    min_dists = []
    for g in tqdm(gen_pcs, desc="MMD"):
        min_dists.append(min(chamfer_distance(g, r) for r in ref_pcs))
    return float(np.mean(min_dists))


def extract_pointnet_features(pcs: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    Global max-pooling over per-point MLPs — permutation invariant.
    pcs: (N, n_points, 3)
    returns: (N, 256) feature vectors
    """
    import torch.nn as nn

    class SimplePointNetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(3, 64),   nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 256)
            )
        def forward(self, x):          # x: (B, N, 3)
            return self.mlp(x).max(1).values   # (B, 256)

    encoder = SimplePointNetEncoder().to(device).eval()
    all_feats = []
    tensor = torch.from_numpy(pcs).float()
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i+batch_size].to(device)
            all_feats.append(encoder(batch).cpu().numpy())
    return np.concatenate(all_feats, axis=0)   # (N, 256)


# ─────────────────────────────────────────────
#  ENTRY POINT FOR METRICS
# ─────────────────────────────────────────────

def compute_all_metrics(gen_mesh_dir: Path, config, device) -> dict:
    print("\n── Loading generated meshes …")
    gen_pcs = load_all_meshes(gen_mesh_dir)
    print(f"   {len(gen_pcs)} generated meshes loaded")

    print("── Loading reference meshes from pkl dataset …")
    ref_pcs = load_reference_pcs_from_dataset(config)
    print(f"   {len(ref_pcs)} reference meshes loaded")

    gen_feats = extract_pointnet_features(gen_pcs, device)
    ref_feats = extract_pointnet_features(ref_pcs, device)

    fid               = compute_fid(gen_feats, ref_feats)
    kid_mean, kid_std = compute_kid(gen_feats, ref_feats)
    mmd               = compute_mmd(gen_pcs, ref_pcs)

    return {"FID": fid, "KID_mean": kid_mean, "KID_std": kid_std, "MMD": mmd}