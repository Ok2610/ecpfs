Search parameters
=================

``new_search`` and ``get_next_k_items`` share four arguments, ``k``,
``search_exp``, ``max_increments`` and ``exclude_vec``. This page explains how
they work together.

How a search runs
-----------------

The index is a tree. A search keeps a queue of nodes to visit, ordered by how
close each node's representative is to the query, and always visits the closest
one next. Visiting a leaf scores every item in it and adds the items to the
query's buffer. The search stops scanning leaves at the limits below, sorts the
buffer, and returns the best ``k`` items. The rest stay buffered for
``get_next_k_items``.

``k``
   How many ``(score, item_id)`` pairs to return. Fewer come back if the search
   finds fewer items.

``search_exp``
   How many leaves to scan in this call. More leaves give better results and a
   slower search. Results are exact within the scanned leaves, so a closer item
   in a leaf that wasn't scanned is missed.

``max_increments``
   What to do when ``search_exp`` leaves gave fewer than ``k`` items. The search
   doubles ``search_exp`` and keeps going, at most ``max_increments`` times.
   ``0`` never doubles, and ``-1`` doubles until ``k`` items are found or the
   whole tree has been scanned.

``exclude_vec``
   Item ids to leave out of the results. It applies to the leaves scanned in this
   call, so an item already buffered by an earlier call can still come back.

Example
-------

With ``k=10``, ``search_exp=4`` and ``max_increments=2``, the search scans 4
leaves. If they hold 10 or more items, it stops there. Otherwise it doubles to 8
leaves in total, then 16, and stops after that even if it still has fewer than
10 items.

Scores
------

Lower is always better. For ``Metric.L2`` the score is the Euclidean distance.
For ``Metric.IP`` it is the negated inner product, so a strong match has a large
negative score.

Paging with ``get_next_k_items``
--------------------------------

``get_next_k_items`` returns the next ``k`` buffered items. If fewer than ``k``
are buffered and unvisited nodes remain, it first scans ``search_exp`` more
leaves, with the same doubling rule. A later page can hold an item that scores
better than one returned earlier, when it comes from a leaf scanned later.

A query that still has results left can be resumed with its id after
``close()``, from a new ``Index`` on the same path (see :doc:`quickstart`).
