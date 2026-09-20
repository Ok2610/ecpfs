Multithreading internals
=========================

This page is for anyone modifying ``ecp-core``'s build or insert code, not
for users of the Python API. It explains how ``build_tree`` and
``Index::insert`` use threads, why they use them differently, and a
deadlock this design has to avoid.

Two pools, not one
-------------------

Nowhere in this codebase do we construct a ``rayon::ThreadPool``. Every use
of ``.into_par_iter()`` runs on rayon's single, lazily-initialized default
pool, sized to the number of CPU cores unless ``RAYON_NUM_THREADS``
overrides it. That pool is shared by two unrelated things:

- Our own code: ``build_tree``'s tree fan-out (``add_data`` and
  ``route_batch_to_node`` in ``build/tree.rs``), which explicitly dispatches
  work via ``.into_par_iter()``.
- ``zarrs``'s own internal chunk parallelism: reading or writing an array
  that spans multiple on-disk chunks uses the ``rayon_iter_concurrent_limit``
  crate internally, targeting up to ``rayon::current_num_threads()`` by
  default. This applies to every array read or write in the codebase,
  including ``Node::embeddings()``/``children()``'s reads during search, and
  ``append_node_batch``'s writes during build or insert.

``zarrs`` does not provide any locking of its own. Its own documentation is
explicit about this: writing the same chunk concurrently from two callers
can lose data, and it is the caller's responsibility to prevent that.
``leaf_locks`` (below) exists to fill exactly that gap.

Why ``build_tree`` and ``Index::insert`` share code but not a threading strategy
----------------------------------------------------------------------------------

Both call the same routing methodology (``add_data``, then recursively
``route_batch_to_node``), which walks the tree from the root down to a
target level and appends there. A ``fan_out`` helper in ``build/tree.rs``
decides, at every level of that walk, whether to dispatch via
``.into_par_iter()`` or run on the calling thread:

.. code-block:: rust

   fn fan_out<F>(count: usize, on_caller_thread: bool, process: F) -> Vec<(u32, u32)>

``build_tree`` always passes ``on_caller_thread = false``, so its fan-out is
parallel at every level. This is safe because a fresh build has
no concurrent readers and no other writer; nothing else is touching the
tree while it's being built.

``Index::insert`` always passes ``on_caller_thread = true``: its own
routing runs entirely on whatever thread called ``insert()``, never
dispatched to rayon. This is required, not a performance choice, and the
next section shows exactly why.

The deadlock this avoids
-------------------------

``Index::insert`` takes a lock on the specific leaf it's about to append to
(``leaf_locks``, a lock per leaf id), so a concurrent search reading that
same leaf can't race the write. If insert's own routing ran on rayon like
``build_tree``'s does, the thread that ends up acquiring that lock could be
a rayon worker thread instead of insert's original caller. That's the
problem. A worker thread that blocks on a lock stops being available for
anything else, including work a concurrent reader is waiting on to finish.

With a single rayon worker, running insert's routing on that worker
reproduces a deadlock. The worker gets stuck on the lock the search
thread holds, so it never returns to run the chunk-read job the search
thread is waiting on. This only happens because insert's own routing task
is what occupies the worker. A search running by itself, with no insert
in the picture, never deadlocks this way: its chunk reads are ordinary
rayon jobs that don't wait on any lock, so any free worker finishes them
normally.

This was found via an ``lldb`` thread dump test which deadlocked.
It reproduces with more than one worker too, it just takes
more simultaneous unlucky assignments for every worker to end up stuck at
once, which becomes more likely as insert activity increases.

The fix keeps the mechanism, not the pool it runs on: as long as the
thread reaching "acquire this leaf's lock" is never a rayon worker, it can
block freely without costing the pool anything, since it was never
available to the pool's other work in the first place. ``on_caller_thread``
is what guarantees that for insert.

What ``leaf_locks`` is and isn't for
--------------------------------------

``leaf_locks`` protects two different things:

- Writer-vs-writer: two concurrent inserts appending to the same leaf at
  once. ``zarrs`` explicitly does not protect against this.
- Reader-vs-writer: a search reading a leaf while an insert is mid-append
  to it. Without a lock, the reader could observe the leaf's shape already
  updated to include new rows whose data hasn't been written yet, and
  either panic or read corrupted data.

It is not protecting against concurrent reads or writes of a single
node's own chunks. That's ``zarrs``'s own internal parallelism, described
above, and it's already safe on its own.

A cache hit never touches ``leaf_locks`` at all. A cache miss at the leaf
level takes the lock only for the duration of the disk read or write
itself, never longer.

A future direction is to split insert into a routing phase (fully parallel)
and a writing phase (parallel with ``leaf_locks``).