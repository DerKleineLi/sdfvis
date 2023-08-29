import trimesh
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.preprocessing import normalize


def load_mesh(path):
    with open(path, "rb") as f:
        mesh = trimesh.load_mesh(f, "ply")
    points = mesh.vertices
    tree = KDTree(points)
    return mesh, tree


def compute_udf(poses, tree):
    distances, _ = tree.query(poses)
    return distances


def compute_sdf(poses, mesh, tree):
    ray_origins = poses[[0], :]
    ray_ends = poses[[-1], :]
    ray_directions = normalize(ray_ends - ray_origins)

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions
    )

    pose_distances = np.linalg.norm(poses - ray_origins, axis=1)
    intersect_distances = np.linalg.norm(locations - ray_origins, axis=1)

    is_outside = np.ones(len(poses), dtype=bool)
    for d in intersect_distances:
        is_outside[pose_distances > d] = ~is_outside[pose_distances > d]

    distances = compute_udf(poses, tree)
    distances[~is_outside] *= -1

    return distances
