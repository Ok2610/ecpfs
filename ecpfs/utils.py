from enum import Enum
from multiprocessing import cpu_count
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from threadpoolctl import threadpool_limits
import zarr
from zarr.storage import ZipStore
import h5py
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.metrics.pairwise import cosine_similarity
import json

# Module‑level globals (set in init_worker)
GLOBAL_D_ZARR = None
GLOBAL_D_IDXS = None
GLOBAL_TMP_ZARR = None
GLOBAL_R_ARRAY = None


class Metric(Enum):
    L2 = 0
    IP = 1
    COS = 2


def calculate_chunk_size(
    num_features: int, dtype: np.dtype, max_chunk_size=50 * 1024 * 1024
) -> Tuple[int, int]:
    """
    Calculates the maximum number of rows that keeps the chunk size below the given limit.

    Parameters:
        num_features (int): Number of features (columns) in the dataset.
        dtype (numpy dtype): Data type of the array.
        max_chunk_size (int): Maximum chunk size in bytes (default: 50 MB).

    Returns:
        tuple: Optimal chunk size (rows, columns).
    """
    bytes_per_value = np.dtype(dtype).itemsize  # Size of one value in bytes
    bytes_per_row = num_features * bytes_per_value  # Size of one row in bytes

    max_rows = max_chunk_size // bytes_per_row  # Max rows per chunk

    if max_rows < 1:
        raise ValueError(
            "Chunk size too small to fit even one row. Increase max_chunk_size or reduce dimensions."
        )

    return (max_rows, num_features)


def get_source_embeddings(
    embeddings_file: Path, grp_name: str
) -> zarr.Array | h5py.Dataset:
    """
    Get the source embeddings from the specified file.
    Parameters:
        embeddings_file (Path): Path to the embeddings file.
        grp (bool): Whether to treat the file as a group or not.
        grp_name (str): Name of the group or dataset to access.
    Returns:
        zarr.Array or h5py.Dataset: The embeddings data.
    """
    embeddings = None
    if embeddings_file.suffix == ".h5":
        embeddings = h5py.File(embeddings_file, "r")[grp_name]
    elif embeddings_file.suffix == ".zarr":
        embeddings = zarr.open(embeddings_file, mode="r")[grp_name]
    elif (
        embeddings_file.suffix == ".zip"
        or str.lower(embeddings_file.suffix) == ".zipstore"
    ):
        embeddings = zarr.open(ZipStore(embeddings_file), mode="r")[grp_name]
    else:
        raise ValueError("Unknown embeddings file format")

    if isinstance(embeddings, zarr.Array) or isinstance(embeddings, h5py.Dataset):
        return embeddings
    else:
        raise ValueError(
            "Group name resulted in a group object. Please provide a dataset/array name."
        )


def calculate_distances(
    q_emb: np.ndarray, embeddings: np.ndarray, metric: Metric
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distance to the embeddings of the node at the specified level

    Parameters:
        q_emb (np.ndarray): The query embedding.
        embeddings (np.ndarray): The embeddings to compare against.
        metric (Metric): The distance metric to use.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays:
            - top: The indices of the sorted distances (argsorted).
            - distances: The calculated distances to the node embeddings.
    """
    if metric == Metric.IP:
        distances = np.dot(embeddings, q_emb)
        top = np.argsort(distances)[::-1]
    elif metric == Metric.L2:
        differences = embeddings - q_emb
        distances = np.linalg.norm(differences)
        top = np.argsort(distances)
    elif metric == Metric.COS:
        distances = cosine_similarity(embeddings, (q_emb,)).flatten()
        top = np.argsort(distances)[::-1]
    return top, distances


def calculate_distances_get_top_k(
    q_emb: np.ndarray, embeddings: np.ndarray, metric: Metric, k: int = 256
) -> np.ndarray:
    """
    Calculate the distance to the embeddings of the node at the specified level

    Parameters:
        q_emb (np.ndarray): The query embedding.
        embeddings (np.ndarray): The embeddings to compare against.
        metric (Metric): The distance metric to use.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays:
            - top: The indices of the sorted distances (argsorted).
            - distances: The calculated distances to the node embeddings.
    """
    if metric == Metric.IP:
        distances = np.dot(embeddings, q_emb)
        return np.argsort(distances)[::-1][:k]
    elif metric == Metric.L2:
        differences = embeddings - q_emb
        distances = np.linalg.norm(differences)
        return np.argsort(distances)[:k]
    elif metric == Metric.COS:
        distances = cosine_similarity(embeddings, (q_emb,)).flatten()
        return np.argsort(distances)[::-1][:k]
    raise ValueError("Unknown metric type")


def determine_node_assignments(
    node_embeddings: np.ndarray,
    data_embeddings: np.ndarray,
    workers: int,
    metric: Metric = Metric.IP,
) -> Tuple[np.ndarray, np.ndarray]:
    with threadpool_limits(limits=workers):
        if metric == Metric.IP:
            S = node_embeddings.dot(data_embeddings.T)
            best_ids = S.argmax(axis=0)
        elif metric == Metric.L2:
            S = node_embeddings - data_embeddings
            max_values = np.linalg.norm(S, axis=1)
            best_ids = np.argmin(max_values, axis=0)
        elif metric == Metric.COS:
            S = cosine_similarity(node_embeddings, data_embeddings.T)
            best_ids = S.argmax(axis=0)
        else:
            raise ValueError("Unknown metric type")
        return group_by_assignments(
            num_reps=node_embeddings.shape[0],
            arr=best_ids,
        )


def group_by_assignments(
    num_reps: int, arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      offsets: uint32 array of shape (num_reps+1,)
      data:    uint32 array of length = sum(counts)

    For any rep r, its members ids are data[offsets[r]:offsets[r+1]]
    from the provided embeddings array.
    """

    counts = np.zeros(num_reps, dtype=np.uint32)
    # bincount up to num_reps
    bc = np.bincount(arr, minlength=num_reps).astype(np.uint32)
    counts += bc

    # Build offsets via prefix-sum
    offsets = np.zeros(num_reps + 1, dtype=np.uint32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)

    # Allocate output arrays and cursors
    data = np.zeros(offsets[-1], dtype=np.uint32)
    cursor = offsets[:-1].copy()

    # Second pass: fill in the indices
    for idx, rep in enumerate(arr):
        pos = cursor[rep]
        data[pos] = np.uint32(idx)
        cursor[rep] += 1

    return offsets, data


def get_memory_size(ds_size: int, nd_size: int, dim_size: int, bsize: int) -> int:
    """
    Calculate the memory size of the embeddings.

    Parameters:
        ds_size (int): Size of the data set.
        nd_size (int): Size of the node set.
        dim_size (int): Dimension size of the embeddings.
        bsize (int): Batch size.

    Returns:
        int: Total memory size in bytes.
    """
    return (
        (ds_size * dim_size * bsize)
        + (nd_size * dim_size * bsize)
        + (ds_size * nd_size * bsize)
    )
