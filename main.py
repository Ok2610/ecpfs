import sys
import typer
from loguru import logger
from typing import Annotated
from pathlib import Path

from ecpfs import ECPBuilder
from ecpfs.ECPBuilder import Metric

app = typer.Typer()
logger.remove()  # Remove default logging handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)


@app.command("build-index")
def build_index(
    embeddings_file: Annotated[
        Path,
        typer.Argument(help="Embeddings file with data vectors. Zarr or HDF5 file"),
    ],
    save_file: Annotated[
        str, typer.Option("--save-file", help="Output file")
    ] = "ecpfs_index.zarr",
    levels: Annotated[int, typer.Option(help="Levels in the index")] = 3,
    target_cluster_items: Annotated[
        int,
        typer.Option(
            "--target-cluster-items",
            help="Preferred items for each cluster (no guarantees)",
        ),
    ] = 100,
    metric: Annotated[
        str,
        typer.Option(
            help="Metric to use for distance calculations. Options: L2 (default) | IP | cos",
            case_sensitive=False,
            metavar="FORMAT",
            show_choices=["L2", "IP", "cos"],
        ),
    ] = "L2",
    workers: Annotated[
        int,
        typer.Option(help="Number of threads involved (default = 4)"),
    ] = 4,
    no_emb_grp: Annotated[
        bool,
        typer.Option(
            "--no-emb-grp",
            help="Indicates if the embeddings dataset is in a group in the file/store",
        ),
    ] = False,
    emb_grp_name: Annotated[
        str,
        typer.Option("--emb-grp-name", help="Group name for the embeddings dataset"),
    ] = "embeddings",
    file_store: Annotated[
        str,
        typer.Option(
            "--file-store",
            help="File store format. Options are 'zarr_l' (zarr.storage.LocalStore), 'zarr_z' (zarr.storage.ZipStore)",
            case_sensitive=False,
            metavar="FORMAT",
            show_choices=["zarr_l", "zarr_z"],
        ),
    ] = "zarr_l",
    rep_selection: Annotated[
        str,
        typer.Option(
            "--rep-selection",
            help="Parameter for how the representatives are selected. Options are 'offset', 'random', 'mbk' (MiniBatchKmeans) or 'dissimilar' (Not implemented)",
            case_sensitive=False,
            metavar="REPSEL",
            show_choices=["offset", "random", "mbk", "dissimilar"],
        ),
    ] = "offset",
    rep_file: Annotated[
        Path,
        typer.Option(
            "--rep-file",
            help="A zarr or HDF5 file containing pre-selected representatives",
        ),
    ] = None,
    rep_emb_grp: Annotated[
        str,
        typer.Option("--rep-emb-grp", help="Representative file embeddings group name"),
    ] = "rep_embeddings",
    rep_ids_grp: Annotated[
        str, typer.Option("--rep-ids-grp", help="Representative file ids group name")
    ] = "rep_item_ids",
    rep_file_store: Annotated[
        str,
        typer.Option(
            "--rep-file-store",
            help="Represenatives file store format. Options are 'zarr_l' (zarr.storage.LocalStore), 'zarr_z' (zarr.storage.ZipStore)",
            case_sensitive=False,
            metavar="FORMAT",
            show_choices=["zarr_l", "zarr_z"],
        ),
    ] = "zarr_l",
) -> None:
    if not embeddings_file.exists():
        logger.error("Embeddings file does not exist!")
        exit(-1)

    _metric = None
    if metric == Metric.L2.name:
        _metric = Metric.L2
    elif metric == Metric.IP.name:
        _metric = Metric.IP
    elif metric == Metric.COS.name:
        _metric = Metric.COS
    else:
        logger.error("Invalid metric")
        exit(-1)

    ecp = ECPBuilder.ECPBuilder(
        levels=levels,
        logger=logger,
        target_cluster_items=target_cluster_items,
        metric=_metric,
        index_file=save_file,
        file_store=file_store,
        workers=workers,
    )

    if rep_file is None:
        logger.info(f"Selecting cluster representatives using {rep_selection}...")
        ecp.select_cluster_representatives(
            embeddings_file=embeddings_file,
            grp=False if no_emb_grp else True,
            grp_name=emb_grp_name,
            option=rep_selection,
        )
    else:
        logger.info("Loading cluster representatives...")
        ecp.get_cluster_representatives_from_file(
            rep_file,
            emb_dsname=rep_emb_grp,
            ids_dsname=rep_ids_grp,
            format=rep_file_store,
        )

    logger.info("Building tree...")
    ecp.build_tree_fs()

    logger.info("Adding items to index...")
    ecp.add_items_concurrent(
        embeddings_file=embeddings_file,
        grp=False if no_emb_grp else True,
        grp_name=emb_grp_name,
    )

    logger.info("Index created")

    return


if __name__ == "__main__":
    app()
