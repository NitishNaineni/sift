import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import struct

idx = 204
image = cv2.imread(f"data/sidewalk/images/{idx}.png")
depth = np.load(f"data/sidewalk/depths/{idx}.npy")

with open("data/sidewalk/metadata.json") as f:
    meta = json.load(f)

T = np.array(meta["transformArray"]).squeeze(1)[idx].T
K = np.array(meta["intrinsicsArray"]).squeeze(1)[idx].T

H, W, _ = image.shape
Hd, Wd = depth.shape

ju = np.arange(Wd, dtype=np.float32) + 0.5
iv = np.arange(Hd, dtype=np.float32) + 0.5
u, v = np.meshgrid(ju, iv, indexing="xy")
u, v = u * W / Wd, v * H / Hd

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

Xc = (u - cx) / fx * depth
Yc = -(v - cy) / fy * depth
Zc = -depth

cam_point_map = np.stack([Xc, Yc, Zc], axis=-1)
Pc = np.stack([Xc, Yc, Zc, np.ones_like(Zc)], axis=-1)
Pw = (T @ Pc.reshape(-1, 4).T).T[:, :3].reshape(Hd, Wd, 3)
pts = Pw.reshape(-1, 3)


def compute_normal_map(point_map, eps=1e-12):
    Pp = np.pad(point_map, ((1, 1), (1, 1), (0, 0)), mode="edge")
    dx = 0.5 * (Pp[1:-1, 2:] - Pp[1:-1, :-2])
    dy = 0.5 * (Pp[2:, 1:-1] - Pp[:-2, 1:-1])
    n = np.cross(dy, dx)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + eps
    return n


normal_map = compute_normal_map(cam_point_map)


ray_depth_map = np.linalg.norm(cam_point_map, axis=-1)


viwe_dir_map = cam_point_map / ray_depth_map[:, :, np.newaxis]

surface_angle_map = (normal_map * viwe_dir_map).sum(-1)
mask = np.abs(surface_angle_map) <= np.sin(np.deg2rad(10))
# mask[ray_depth_map > 10] = True


def construct_faces(h, w, depth_map):
    faces = np.empty((2 * (h - 1) * (w - 1), 3), dtype=np.int64)

    for r in range(h - 1):
        for c in range(w - 1):
            top_left = r * w + c
            top_right = r * w + (c + 1)
            bottom_left = (r + 1) * w + c
            bottom_right = (r + 1) * w + (c + 1)

            index = 2 * (r * (w - 1) + c)
            if abs(depth_map[r, c] - depth_map[r + 1, c + 1]) > abs(
                depth_map[r + 1, c] - depth_map[r, c + 1]
            ):
                faces[index] = top_left, bottom_left, top_right
                faces[index + 1] = bottom_right, top_right, bottom_left
            else:
                faces[index] = bottom_right, top_left, bottom_left
                faces[index + 1] = top_left, bottom_right, top_right

    return faces


def save_mesh_ply(path, point_map, faces):
    verts = point_map.reshape(-1, 3).astype(np.float32)
    valid = np.isfinite(verts).all(axis=1)

    new_idx = np.full(len(verts), -1, dtype=np.int64)
    new_idx[valid] = np.arange(valid.sum(), dtype=np.int64)

    f = faces.astype(np.int64, copy=False)
    mask_valid = valid[f].all(axis=1)
    f = f[mask_valid]
    f = new_idx[f]  # remap
    mask_distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[mask_distinct]

    v = verts[valid]
    nv, nf = len(v), len(f)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {nv}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {nf}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    with open(path, "wb") as fp:
        fp.write(header)
        fp.write(v.tobytes(order="C"))
        for tri in f.astype(np.int32, copy=False):
            fp.write(struct.pack("<Biii", 3, tri[0], tri[1], tri[2]))

    return nv, nf


faces = construct_faces(Hd, Wd, ray_depth_map)

face_mask = np.any(mask.flatten()[faces], axis=-1)

faces = faces[~face_mask]


save_mesh_ply("mesh.ply", Pw, faces)


np.savetxt("points.xyz", pts, fmt="%.6f")
