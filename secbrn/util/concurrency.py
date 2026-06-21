"""Bounded, order-preserving concurrency for I/O-bound stages.

The model providers (Ollama) talk over a blocking ``httpx.Client``; every embed /
generate is a network round-trip that spends almost all its wall-clock time waiting
on a socket while holding no useful CPU. The single biggest speed-up for ingest is
therefore to have several of those calls *in flight at once* instead of strictly one
after another.

A thread pool — not ``asyncio`` — is the right tool here:

* ``httpx.Client`` is synchronous and thread-safe; blocking socket reads release the
  GIL, so threads get real overlap for this exact workload.
* Going async would mean rewriting every provider, store method, the CLI and the
  tests around ``await``. Threads give the same wall-clock win with a localized,
  low-risk change and identical, deterministic results.

:func:`map_workers` is deliberately tiny and result-shaped like a sequential loop:
it returns one ``(result, exception)`` per input **in input order**, so callers keep
the per-item error handling (and reporting order) they had before — only faster. An
optional ``on_complete`` callback fires once per finished item (on the calling
thread) so callers can drive a progress bar without caring about the threading.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# (result, exception): exactly one is non-None, mirroring a try/except per item.
Outcome = tuple[R | None, Exception | None]


def map_workers(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int,
    on_complete: Callable[[], None] | None = None,
) -> list[Outcome]:
    """Apply ``fn`` to every item, up to ``workers`` calls concurrently.

    Returns a list aligned with ``items``; each element is ``(result, None)`` on
    success or ``(None, exc)`` if ``fn`` raised. Exceptions are captured, never
    propagated, so one bad item can't sink the batch — the caller decides what to do
    with each outcome, just like the sequential ``try/except`` loop it replaces.

    ``on_complete`` (if given) is called exactly once for each item as it finishes,
    on the thread that owns this call (the main thread), so it is safe to update a
    progress bar / counter from it without locking.

    ``workers <= 1`` (or a 0/1-length batch) runs inline with no threads, preserving
    exact sequential semantics for tests and tiny inputs.
    """
    n = len(items)
    if n == 0:
        return []
    if workers <= 1 or n == 1:
        out: list[Outcome] = []
        for it in items:
            out.append(_call(fn, it))
            if on_complete is not None:
                on_complete()
        return out

    results: list[Outcome] = [(None, None)] * n
    with ThreadPoolExecutor(max_workers=min(workers, n)) as pool:
        # Submit by index; write results back by index so output stays in input order
        # regardless of completion order. Drive ``on_complete`` from as_completed so a
        # progress bar advances in real time as calls actually finish.
        futures = {pool.submit(_call, fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()  # _call never raises
            if on_complete is not None:
                on_complete()
    return results


def _call(fn: Callable[[T], R], item: T) -> Outcome:
    try:
        return fn(item), None
    except Exception as e:  # captured, surfaced to the caller as data
        return None, e
