from pathlib import Path
from typing import Tuple
import numpy as np
import zarr
from zarr.storage import ZipStore
import h5py


def calculate_chunk_size(
    num_features: int, dtype: np.dtype, max_chunk_size=50 * 1024 * 1024
) -> Tuple[int, int]:
    """
    Calculates the maximum number of rows that keeps the chunk size below the given limit.

    Args:
        shape (tuple): Shape of the dataset (rows, columns).
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


def get_source_embeddings(embeddings_file: Path, grp: bool, grp_name: str) -> zarr.Array | h5py.Dataset:
    embeddings = None
    if embeddings_file.suffix == ".h5":
        embeddings = h5py.File(embeddings_file, "r")[grp_name]
    elif embeddings_file.suffix == ".zarr":
        if grp:
            embeddings = zarr.open(embeddings_file, mode="r")[grp_name]
        else:
            embeddings = zarr.open_array(embeddings_file, mode="r")
    elif (
        embeddings_file.suffix == ".zip"
        or str.lower(embeddings_file.suffix) == ".zipstore"
    ):
        if grp:
            embeddings = zarr.open(
                ZipStore(embeddings_file, mode="r"), mode="r"
            )[grp_name]
        else:
            embeddings = zarr.open_array(
                ZipStore(embeddings_file, mode="r"), mode="r"
            )
    else:
        raise ValueError("Unknown embeddings file format")

    if isinstance(embeddings, zarr.Array) or isinstance(embeddings, h5py.Dataset):
        return embeddings
    else:
        raise ValueError("Group name resulted in a group object. Please provide a dataset/array name.")