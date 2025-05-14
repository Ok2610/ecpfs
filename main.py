import typer
from typing import Annotated
from pathlib import Path

from ecpfs import Metric, ECPBuilder, ecp_enable_logging
ecp_enable_logging()

app = typer.Typer()

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
    memory_limit: Annotated[
        int,
        typer.Option(
            "--memory-limit",
            help="The amount of memory the build process should adhere to in GB. Not guaranteed. Default = 4 GB",
        ),
    ] = 4,
) -> None:
    if not embeddings_file.exists():
        print("Embeddings file does not exist!")
        exit(-1)

    _metric = None
    if metric == Metric.L2.name:
        _metric = Metric.L2
    elif metric == Metric.IP.name:
        _metric = Metric.IP
    elif metric == Metric.COS.name:
        _metric = Metric.COS
    else:
        print("Invalid metric")
        exit(-1)

    ecp = ECPBuilder(
        levels=levels,
        target_cluster_items=target_cluster_items,
        metric=_metric,
        index_file=save_file,
        file_store=file_store,
        workers=workers,
        memory_limit=memory_limit,
    )

    if rep_file is None:
        ecp.select_cluster_representatives(
            embeddings_file=embeddings_file,
            grp_name=emb_grp_name,
            option=rep_selection,
        )
    else:
        ecp.get_cluster_representatives_from_file(
            rep_file,
            emb_dsname=rep_emb_grp,
            ids_dsname=rep_ids_grp,
            format=rep_file_store,
        )

    ecp.build_tree_fs(
        embeddings_file=embeddings_file,
        grp_name=emb_grp_name,
    )

    return


if __name__ == "__main__":
    app()
