Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install ecpfs

Building an index
------------------

.. code-block:: python

   from ecpfs import Builder, Metric

   builder = Builder(
       index_path="my_index.zarr",
       levels=3,
       metric=Metric.L2,
       is_normalized=False,
   )
   builder.select_representatives(
       embeddings_file="embeddings.zarr",
       target_cluster_items=100,
       strategy="offset",
       fallback_batch_rows=100_000,
   )
   builder.build(
       embeddings_file="embeddings.zarr",
       fallback_batch_rows=100_000,
   )

Searching an index
--------------------

.. code-block:: python

   import numpy as np
   from ecpfs import Index

   index = Index("my_index.zarr")

   query = np.random.rand(128).astype(np.float32)
   items, query_id = index.new_search(
       query=query,
       k=10,
       search_exp=4,
       max_increments=-1,
       exclude_vec=[],
   )
   # items: list[(distance, item_id)]

   # Pull further results for the same query without re-searching from the root:
   more_items = index.incremental_search(
       query_id=query_id,
       k=10,
       search_exp=4,
       max_increments=-1,
       exclude_vec=[],
   )

Enabling logging
------------------

.. code-block:: python

   from ecpfs import init_logging

   log_path = init_logging(log_dir="ecp_logs", level="debug")
