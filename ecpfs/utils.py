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


def init_worker(d_zarr_path, d_idxs, tmp_zarr_path, r_array):
    # Pin BLAS to 1 thread before any dot() calls
    os.environ["OMP_NUM_THREADS"] = "1"
    global GLOBAL_D_ZARR, GLOBAL_D_IDXS, GLOBAL_R_ARRAY, GLOBAL_TMP_ZARR
    GLOBAL_D_ZARR = d_zarr_path
    GLOBAL_D_IDXS = d_idxs
    GLOBAL_R_ARRAY = r_array
    GLOBAL_TMP_ZARR = tmp_zarr_path


def node_worker(d_start: int, d_end: int, metric: Metric = Metric.IP):
    """
    Worker function to compute the best assignments for a range of data embeddings.
    Avoids pickling the zarr.Array by using a global variable.
    """
    # Get embeddings from file
    D_zarr = get_source_embeddings(GLOBAL_D_ZARR[0], GLOBAL_D_ZARR[1])
    if GLOBAL_D_IDXS is not None:
        D = D_zarr[GLOBAL_D_IDXS[d_start:d_end]]
    else:
        D = D_zarr[d_start:d_end]
    D = np.squeeze(D)

    if metric == Metric.IP:
        S = GLOBAL_R_ARRAY.dot(D.T)
        best_ids = S.argmax(axis=0).astype(np.uint32)
    elif metric == Metric.L2:
        S = GLOBAL_R_ARRAY - D
        max_values = np.linalg.norm(S, axis=1)
        best_ids = np.argmin(max_values, axis=0).astype(np.uint32)
    elif metric == Metric.COS:
        S = cosine_similarity(GLOBAL_R_ARRAY, D.T)
        best_ids = S.argmax(axis=0).astype(np.uint32)
    else:
        raise ValueError("Unknown metric type")

    zarr.open(GLOBAL_TMP_ZARR, mode="a")[d_start:d_end] = best_ids


def determine_node_assignments(
    tmp_zarr_pth: str,
    node_embeddings: np.ndarray,
    data_emb_path: Tuple[str, str],  # (file, group)
    data_embeddings: zarr.Array,
    workers: int,
    mem_threshold: int,
    mem_limit: int,
    est_mem_size: int,
    arr_name: str = "data_ids",
    metric: Metric = Metric.IP,
    data_idxs: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ds_size = len(data_idxs)

    if est_mem_size <= mem_threshold:
        with threadpool_limits(limits=1):
            embeddings = np.squeeze(data_embeddings[data_idxs])
            if metric == Metric.IP:
                S = node_embeddings.dot(embeddings.T)
                best_ids = S.argmax(axis=0)
            elif metric == Metric.L2:
                S = node_embeddings - embeddings
                max_values = np.linalg.norm(S, axis=1)
                best_ids = np.argmin(max_values, axis=0)
            elif metric == Metric.COS:
                S = cosine_similarity(node_embeddings, embeddings.T)
                best_ids = S.argmax(axis=0)
            else:
                raise ValueError("Unknown metric type")
            return group_by_assignments(
                zarr_path="",
                num_reps=node_embeddings.shape[0],
                arr=best_ids,
            )

    memory_batches = 1
    # Calculate the number of batches needed to fit in memory if exceeding the limit
    if est_mem_size > mem_limit:
        memory_batches = math.ceil(est_mem_size / mem_limit)
    # Number of data embeddings per batch
    memory_batch_size = math.floor(ds_size / memory_batches)

    # Number of data embeddings each worker will process
    w_batch_size = math.ceil(memory_batch_size / workers)
    grp = zarr.open(tmp_zarr_pth, mode="a")
    grp.create_array(
        name=arr_name, shape=(ds_size,), chunks=(w_batch_size,), dtype=np.uint32
    )

    process_futures = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            data_emb_path,
            data_idxs,
            f"{tmp_zarr_pth}/{arr_name}",
            node_embeddings,
        ),
    ) as exe:
        for m_batch in range(memory_batches):
            mb_start = m_batch * memory_batch_size
            mb_end = min((m_batch + 1) * memory_batch_size, ds_size)
            for d_start in range(mb_start, mb_end, w_batch_size):
                d_end = min(d_start + w_batch_size, mb_end)
                process_futures.append(exe.submit(node_worker, d_start, d_end))
        
        for fut in as_completed(process_futures):
            try:
                fut.result()
            except Exception as e:
                raise Exception(f"Failed in determine_node_assignments with {e}")
                

        exe.shutdown(wait=True)

    offsets, data = group_by_assignments(
        zarr_path=f"{tmp_zarr_pth}/{arr_name}", num_reps=node_embeddings.shape[0]
    )
    remove_zarr_arr(zarr_path=tmp_zarr_pth, array_name=arr_name)
    return offsets, data


def group_by_assignments(
    zarr_path: str, num_reps: int, arr: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the 1D Zarr array at `zarr_path/array_name` (shape N),
    where each element is an integer in [0..num_reps-1] indicating
    the representative assignment for that data point.

    Returns:
      offsets: uint32 array of shape (num_reps+1,)
      data:    uint32 array of length = sum(counts)

    For any rep r, its members are data[offsets[r]:offsets[r+1]].
    """
    if arr is None:
        # Open the Zarr store and array
        ids = zarr.open(zarr_path, mode="r")
        chunk = ids.chunks[0]
    else:
        ids = arr
        chunk = ids.shape[0]
    N = ids.shape[0]

    # First pass: count assignments per rep
    counts = np.zeros(num_reps, dtype=np.uint32)
    for start in range(0, N, chunk):
        sub = ids[start : start + chunk]  # lazy load ≤ chunk elements
        # bincount up to num_reps
        bc = np.bincount(sub, minlength=num_reps).astype(np.uint32)
        counts += bc

    # Build offsets via prefix-sum
    offsets = np.zeros(num_reps + 1, dtype=np.uint32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)

    # Allocate output arrays and cursors
    data = np.zeros(offsets[-1], dtype=np.uint32)
    cursor = offsets[:-1].copy()

    # Second pass: fill in the indices
    for start in range(0, N, chunk):
        sub = ids[start : start + chunk]
        for idx, rep in enumerate(sub):
            pos = cursor[rep]
            data[pos] = np.uint32(start + idx)
            cursor[rep] += 1

    return offsets, data


def remove_zarr_arr(zarr_path: str, array_name: str = "data_ids"):
    """
    Remove the specified Zarr array from the store.
    Parameters:
        zarr_path (str): Path to the Zarr store.
        array_name (str): Name of the array to remove.
    """
    g = zarr.open(zarr_path, mode="a")
    if array_name in g:
        del g[array_name]


def pick_thread_proc_split(workers: int):
    """
    Given a user‐requested total parallelism `workers`, pick
    (T_threads, P_procs_per_thread) so that T * P <= physical cores.
    We try to keep T * P as close to `workers` as possible.
    """
    n_cores = cpu_count() - 1  # leave one core free for the OS

    # start with the naive split
    T = max(1, workers // 2)
    P = max(1, math.floor(workers / T))

    # if we’re already within the budget, great
    if T * P <= n_cores:
        return T, P

    # otherwise bump P down until T*P fits, or if that bottoms out
    # reduce T instead
    while T * P > n_cores and (T > 1 or P > 1):
        if P > 1 and (P >= T or T == 1):
            P -= 1
        else:
            T -= 1
        # recompute to stay as close as possible
        if T > 0:
            P = min(P, max(1, math.floor(workers / T)))

    return T, P


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
