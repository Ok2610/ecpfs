from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from . import ecp as ecp

class Metric(Enum):
    L2 = ...
    IP = ...

class EmbeddingDtype(Enum):
    F16 = ...
    F32 = ...

class Index:
    def __init__(self, index_path: Path | str, memory_limit_bytes: int = ...) -> None: ...
    def set_memory_limit_bytes(self, memory_limit_bytes: int) -> None: ...
    def new_search(
        self,
        query: NDArray[np.float32],
        k: int,
        search_exp: int,
        max_increments: int,
        exclude_vec: Sequence[int],
    ) -> tuple[list[tuple[float, int]], int]: ...

    def get_next_k_items(
        self,
        query_id: int,
        k: int,
        search_exp: int,
        max_increments: int,
        exclude_vec: Sequence[int],
    ) -> list[tuple[float, int]]: ...

    def insert(self, embeddings: NDArray[np.float32]) -> tuple[int, int]: ...
    def cleanup_persisted_queries_older_than(self, cutoff_unix_secs: float) -> int: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Index": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

class Builder:
    def __init__(
        self,
        index_path: Path | str,
        levels: int,
        metric: Metric,
        is_normalized: bool = ...,
        memory_limit_bytes: int = ...,
        embedding_dtype: EmbeddingDtype | None = ...,
        max_chunk_bytes: int = ...,
    ) -> None: ...

    def select_representatives(
        self,
        embeddings_file: Path | str,
        target_cluster_items: int,
        strategy: str,
        fallback_batch_rows: int,
        grp_name: str = ...,
    ) -> None: ...

    def select_representatives_custom(
        self, ids: NDArray[np.uint32], embeddings: NDArray[np.float32]
    ) -> None: ...

    def build(
        self, embeddings_file: Path | str, fallback_batch_rows: int, grp_name: str = ...
    ) -> None: ...

def init_logging(log_dir: str | None = ..., level: str = ...) -> str: ...
