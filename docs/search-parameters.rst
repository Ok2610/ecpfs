Search Parameters
=================

``new_search`` and ``get_next_k_items`` share four arguments, ``k``,
``search_exp``, ``max_increments`` and ``exclude_vec``. ``new_search`` also
takes ``strategy``, which sets the order the tree's nodes are visited in. This
page explains how they work together.

How a search runs
-----------------

The index is a tree. Visiting a node scores its children against the query.
Visiting a leaf node scores every item in it and adds the items to the query's
buffer. ``strategy`` decides which nodes are visited.

``"level_beam"`` (the default)
   The search visits the ``search_exp`` best nodes on each level, from the root
   down to the leaf nodes. A later call starts at the deepest level that still
   holds ``search_exp`` nodes waiting to be visited.

``"best_first"``
   The search keeps one queue of nodes from every level, ordered by how close
   each node's representative is to the query, and always visits the closest
   one next.

``level_beam`` usually reaches the same recall faster. On an index with very
uneven clusters the two are close. A query keeps the strategy it was started
with.

The search stops scanning leaf nodes at the limits below, sorts the buffer, and
returns the best ``k`` items. The rest stay buffered for ``get_next_k_items``.

``k``
   How many ``(score, item_id)`` pairs to return. Fewer come back when the
   search does not find that many, whether because too few items were
   scanned or because ``exclude_vec`` filtered some out.

``search_exp``
   How many leaf nodes to scan in this call. With ``level_beam`` it is also how
   many nodes to visit on every level above them. More leaf nodes give better
   results and a slower search. Results are exact within the scanned leaf nodes,
   so a closer item in a leaf node that wasn't scanned is missed. Must be at
   least 1.

``max_increments``
   What to do when ``search_exp`` leaf nodes gave fewer than ``k`` items. The
   search doubles ``search_exp`` and keeps going, at most ``max_increments``
   times. ``0`` never doubles, and ``-1`` doubles until ``k`` items are found
   or the whole tree has been scanned.

``exclude_vec``
   Item ids to leave out of the results. The query keeps them as its exclude
   set, also when it is saved and resumed, so ``get_next_k_items`` only needs
   ids to add to the set. Added ids are also removed from the items found but
   not yet returned.

Example
-------

With ``k=10``, ``search_exp=4`` and ``max_increments=2``, the search scans 4
leaf nodes. If they hold 10 or more items, it stops there. Otherwise it doubles
``search_exp`` twice at most and stops after that even if it still has fewer
than 10 items. ``best_first`` counts leaf nodes across the call, so it scans up
to 8 in total, then 16. ``level_beam`` descends again after each doubling, so it
scans up to 8 more, then 16 more.

Scores
------

Lower is always better. For ``Metric.L2`` the score is the Euclidean distance.
For ``Metric.IP`` it is the negated inner product, so a strong match has a large
negative score.

Paging with ``get_next_k_items``
--------------------------------

``get_next_k_items`` returns the next ``k`` buffered items. If fewer than ``k``
are buffered and unvisited nodes remain, it first scans ``search_exp`` more
leaf nodes, with the same doubling rule. A later page can hold an item that
scores better than one returned earlier, when it comes from a leaf node scanned
later.

A query is saved to disk when ``close()`` runs or when it is evicted from
memory. If it still has results left, a new ``Index`` on the same path can
resume it with its id (see :doc:`quickstart`). A query that was never saved
cannot be resumed.
