import sys
import typer
from loguru import logger
from typing_extensions import Annotated
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py

from ecph5 import ECPBuilder

app = typer.Typer()
logger.remove()  # Remove default logging handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@app.command()
def build_index(
    embeddings_file: Annotated[
        Path,
        typer.Argument(
            help="Embeddings file with data vectors. HDF5 file with dataset key 'embeddings'"
        ),
    ],
    save_file: Annotated[str, typer.Argument(help="Output file")],
    levels: Annotated[int, typer.Option(help="Levels in the index")] = 3,
    target_cluster_size: Annotated[
        int, typer.Option(help="Preferred size of clusters (no guarantees)")
    ] = 100,
    metric: Annotated[
        str,
        typer.Option(
            help="Metric to use for distance calculations. Options: L2 (default) | IP | cos"
        ),
    ] = "L2",
    workers: Annotated[
        int,
        typer.Option(help="Number of threads involved (default = 4)"),
    ] = 4,
) -> None:
    if not embeddings_file.exists():
        logger.error("Embeddings file does not exist!")
        exit(-1)
    ecp = ECPBuilder.ECPBuilder(
        levels=levels,
        logger=logger,
        target_cluster_size=target_cluster_size,
        metric=metric,
    )

    logger.info("Selecting cluster representatives...")
    ecp.select_cluster_representatives(
        embeddings_file=embeddings_file, save_to_file=save_file
    )
    logger.info("Building tree...")
    ecp.build_tree_h5(save_to_file=save_file)
    logger.info("Adding items to index...")
    ecp.add_items_concurrent(
        embeddings_file=embeddings_file, save_to_file=save_file, workers=workers
    )
    logger.info("Index created")

    return


if __name__ == "__main__":
    app()
