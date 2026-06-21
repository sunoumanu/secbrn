"""`secbrn` command-line interface — a thin wrapper over :class:`Brain`.

Commands: ingest, ask, search, stats, resolve, reindex, healthcheck.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from secbrn.config import get_settings
from secbrn.graph import resolve_backend
from secbrn.pipeline import Brain

app = typer.Typer(add_completion=False, help="SecBrn — local graph second brain for LLMs.")
console = Console()


def _banner() -> None:
    """Show where data will go — the fastest way to catch 'wrong backend' mistakes."""
    s = get_settings()
    backend = resolve_backend(s)
    target = f"neo4j @ {s.neo4j_uri} (db={s.neo4j_database})" if backend == "neo4j" else "in-memory (NOT persisted)"
    color = "green" if backend == "neo4j" else "yellow"
    console.print(
        f"[dim]provider=[/dim]{s.provider}  "
        f"[dim]store=[/dim][{color}]{backend}[/{color}] → {target}  "
        f"[dim]embed=[/dim]{s.embed_model}({s.embed_dim})"
    )
    if backend == "memory":
        console.print(
            "[yellow]⚠ store is in-memory — data will NOT appear in Neo4j and is lost "
            "when this process exits. Set SECBRN_GRAPH_BACKEND=neo4j.[/yellow]"
        )


def _brain() -> Brain:
    return Brain.from_env()


@app.command()
def ingest(
    path: str = typer.Argument(None, help="File or folder to ingest."),
    url: str = typer.Option(None, "--url", help="Ingest a web page instead of a path."),
    no_resolve: bool = typer.Option(False, "--no-resolve", help="Skip entity resolution."),
    debug: bool = typer.Option(False, "--debug", help="Re-raise the first error with a full traceback."),
):
    """Ingest a file/folder or a URL into the brain (Stages 1–6)."""
    if not path and not url:
        console.print("[red]Provide a PATH or --url[/red]")
        raise typer.Exit(2)
    _banner()
    brain = _brain()
    try:
        rep = (
            brain.ingest_url(url, resolve=not no_resolve)
            if url
            else brain.ingest(path, resolve=not no_resolve, raise_errors=debug)
        )
    finally:
        brain.close()
    failed = rep.chunks_failed + rep.extractions_failed
    fail_str = f" [yellow]chunks_failed={rep.chunks_failed} extractions_failed={rep.extractions_failed}[/yellow]" if failed else ""
    console.print(
        f"[green]Ingested[/green] new={rep.documents_ingested} "
        f"updated={rep.documents_updated} skipped={rep.documents_skipped} "
        f"chunks={rep.chunks_written} entities={rep.entities_extracted} "
        f"relations={rep.relations_extracted} merges={rep.merges}" + fail_str
    )
    errors = [d for d in rep.details if d.startswith("ERROR")]
    for d in rep.details:
        style = "red" if d.startswith("ERROR") else ("yellow" if "failed" in d else "dim")
        console.print(f"  [{style}]{d}[/{style}]")
    if errors and rep.chunks_written == 0:
        console.print(
            "[red]Nothing was written.[/red] Every file errored above — commonly Ollama "
            "is unreachable or the embedding dim mismatches. Run [bold]secbrn doctor[/bold] "
            "or re-run with [bold]--debug[/bold] for a traceback."
        )
    elif failed:
        console.print(
            f"[yellow]Done with {failed} chunk-level failure(s)[/yellow] (skipped, rest ingested). "
            "Often a slow model timing out — raise SECBRN_OLLAMA_EMBED_TIMEOUT / "
            "SECBRN_OLLAMA_LLM_TIMEOUT, then re-run `secbrn ingest` (idempotent) to fill gaps."
        )
    elif rep.documents_ingested == 0 and rep.documents_updated == 0 and rep.documents_skipped:
        console.print("[yellow]All files were skipped as unchanged (idempotent dedup).[/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="A question for the brain."),
    top_k: int = typer.Option(None, "--top-k", help="Chunks to retrieve."),
    hops: int = typer.Option(None, "--hops", help="Graph-expansion hops."),
):
    """Ask a multi-hop question; get a cited answer (Stages 7–8)."""
    brain = _brain()
    try:
        ans = brain.ask(question, top_k=top_k, hops=hops)
    finally:
        brain.close()
    console.print(f"\n{ans.text}\n")
    if ans.uncited:
        console.print("[yellow]⚠ answer contained no inline citation — treat with caution[/yellow]")
    if ans.citations:
        console.print("[bold]Sources[/bold]")
        for c in ans.citations:
            console.print(f"  {c.marker} {c.document_title} — {c.span}  [dim]{c.uri}[/dim]")
    if ans.bundle and ans.bundle.edges:
        console.print("\n[bold]Subgraph used[/bold]")
        for e in ans.bundle.edges[:12]:
            console.print(f"  [dim]{e.subject} -[{e.relation}]-> {e.object}[/dim]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Hybrid (vector + full-text) chunk search."),
    top_k: int = typer.Option(None, "--top-k"),
):
    """Retrieve raw chunks without synthesis."""
    brain = _brain()
    try:
        hits = brain.search(query, top_k=top_k)
    finally:
        brain.close()
    for rc in hits:
        console.print(f"[cyan]{rc.score:.3f}[/cyan] [{rc.via}] {rc.document_title} ({rc.span.cite()})")
        console.print(f"  [dim]{rc.text[:200]}{'…' if len(rc.text) > 200 else ''}[/dim]")


@app.command()
def stats():
    """Graph statistics: node/edge counts, orphan chunks, merges."""
    brain = _brain()
    try:
        s = brain.stats()
    finally:
        brain.close()
    table = Table(title="SecBrn graph stats")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in s.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def resolve():
    """Run the entity-resolution pass (Stage 6) on demand."""
    brain = _brain()
    try:
        decisions = brain.resolve()
    finally:
        brain.close()
    console.print(f"[green]Merged {len(decisions)} duplicate entities[/green]")
    for d in decisions:
        es = f"{d.embed_score:.2f}" if d.embed_score is not None else "—"
        console.print(f"  [dim]{d.duplicate} → {d.canonical}  (str={d.string_score:.2f} emb={es} {d.reason})[/dim]")


@app.command()
def reindex(
    recreate: bool = typer.Option(False, "--recreate", help="Drop & recreate the vector index at the configured SECBRN_EMBED_DIM (use after changing the embedding model)."),
):
    """Re-apply schema DDL (constraints + indexes); optionally recreate the vector index."""
    brain = _brain()
    try:
        brain.reindex(recreate=recreate)
    finally:
        brain.close()
    if recreate:
        console.print("[green]Vector index recreated at the configured dimension.[/green]")
        console.print("[yellow]Now re-ingest your sources to re-embed at the new dimension.[/yellow]")
    else:
        console.print("[green]Schema/indexes ensured.[/green]")


@app.command()
def healthcheck():
    """Verify Neo4j + models reachable and indexes present (Phase 0 exit)."""
    from secbrn.healthcheck import run

    raise typer.Exit(run())


@app.command()
def doctor():
    """Diagnose 'no data' problems: show config, backend, connectivity, and counts."""
    s = get_settings()
    backend = resolve_backend(s)
    _banner()

    if backend == "memory":
        console.print(
            "\n[yellow]You are on the in-memory backend.[/yellow] This never writes to "
            "Neo4j. To persist, set [bold]SECBRN_GRAPH_BACKEND=neo4j[/bold] (and have "
            "Neo4j running), then re-ingest."
        )
        raise typer.Exit(1)

    # Probe Neo4j directly so we separate 'can't connect' from 'connected but empty'.
    from secbrn.graph.neo4j_store import Neo4jStore

    store = Neo4jStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password, s.neo4j_database, s.embed_dim)
    try:
        if not store.ping():
            console.print(
                f"[red]✗ cannot reach Neo4j at {s.neo4j_uri}[/red] — is the container up "
                "(`docker compose up -d`) and are the credentials in .env correct?"
            )
            raise typer.Exit(1)
        console.print(f"[green]✓ connected to Neo4j[/green] ({s.neo4j_uri}, db={s.neo4j_database})")
        idx = store.indexes_present()
        for name, present in idx.items():
            mark = "[green]✓[/green]" if present else "[red]✗ missing[/red]"
            console.print(f"  {mark} index {name}")
        if not all(idx.values()):
            console.print("  [yellow]→ run `secbrn reindex` to create missing indexes.[/yellow]")
        st = store.stats()
        table = Table(title="Neo4j contents")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for k, v in st.items():
            table.add_row(k, str(v))
        console.print(table)
        if st.get("documents", 0) == 0:
            console.print(
                "\n[yellow]Connected, but the graph is empty.[/yellow] Likely causes: you "
                "ingested while on the memory backend, ingestion errored (run "
                "`secbrn ingest <path> --debug`), or you're viewing a different database "
                f"in Neo4j Browser than '{s.neo4j_database}'."
            )
        sample_dim = store.sample_chunk_embed_dim()
        if sample_dim is not None and sample_dim != s.embed_dim:
            console.print(
                f"\n[red]✗ embedding-dim mismatch:[/red] chunks were embedded at "
                f"{sample_dim} dims but SECBRN_EMBED_DIM={s.embed_dim}. You changed the "
                f"embedding model — run [bold]secbrn reindex --recreate[/bold] then re-ingest."
            )
        elif sample_dim is not None:
            console.print(f"  [green]✓[/green] embedding dim {sample_dim} matches config")
    finally:
        store.close()


@app.command("eval")
def eval_cmd(
    gold: str = typer.Option("eval/gold.json", "--gold", help="Path to the gold-set JSON/YAML."),
    k: int = typer.Option(None, "--k", help="Cutoff k for retrieval metrics (default: retrieve_top_k)."),
    use_existing: bool = typer.Option(False, "--use-existing", help="Evaluate against the current store; do NOT ingest the gold corpus."),
    show_cases: bool = typer.Option(False, "--show-cases", help="Print per-query retrieval detail."),
):
    """Measure correctness: retrieval / extraction / resolution metrics vs. a gold set."""
    from secbrn.eval import Evaluator, load_goldset

    gs = load_goldset(gold)
    _banner()
    # Isolate retrieval eval: ingest the gold corpus into a throwaway in-memory brain
    # so unrelated data in the main store can't dilute precision. --use-existing scores
    # against the live store instead.
    if use_existing:
        brain = _brain()
        console.print("[dim]evaluating against the existing store (--use-existing)[/dim]")
    else:
        brain = Brain.isolated()
        if gs.retrieval:
            cp = gs.corpus_path()
            if cp is None:
                console.print("[yellow]No corpus in gold set; retrieval will be empty in isolated mode.[/yellow]")
            else:
                rep = brain.ingest(cp)
                console.print(f"[dim]isolated eval corpus {cp} (chunks={rep.chunks_written}, merges={rep.merges})[/dim]")
    try:
        report = Evaluator(brain, k=k).evaluate(gs)
    finally:
        brain.close()

    def _prf_row(table, name, prf):
        table.add_row(name, f"{prf.precision:.2f}", f"{prf.recall:.2f}", f"{prf.f1:.2f}",
                      str(prf.tp), str(prf.fp), str(prf.fn))

    if report.retrieval:
        r = report.retrieval
        t = Table(title=f"Retrieval (k={r.k}, n={r.n})")
        for col in ("precision@k", "precision@R", "recall@k", "MAP", "nDCG@k", "MRR", "hit@k"):
            t.add_column(col, justify="right")
        t.add_row(f"{r.precision_at_k:.2f}", f"{r.r_precision:.2f}", f"{r.recall_at_k:.2f}",
                  f"{r.map:.2f}", f"{r.ndcg_at_k:.2f}", f"{r.mrr:.2f}", f"{r.hit_at_k:.2f}")
        console.print(t)
        console.print("[dim]precision@R and nDCG@k are the reliable ones when relevant<<k.[/dim]")
        if show_cases:
            for c in r.per_case:
                ok = "[green]✓[/green]" if c["rr"] > 0 else "[red]✗[/red]"
                console.print(f"  {ok} P@R={c['r_precision']:.2f} nDCG={c['ndcg']:.2f} "
                              f"AP={c['ap']:.2f} RR={c['rr']:.2f}  [dim]{c['query']}[/dim]")
                console.print(f"      got={c['retrieved']}  want={c['relevant']}")

    if report.entities or report.triples:
        t = Table(title="Extraction (micro-averaged)")
        for col in ("task", "precision", "recall", "F1", "TP", "FP", "FN"):
            t.add_column(col, justify="right")
        if report.entities:
            _prf_row(t, "entities", report.entities)
        if report.triples:
            _prf_row(t, "triples", report.triples)
        console.print(t)

    if report.resolution:
        t = Table(title="Resolution (pairwise)")
        for col in ("task", "precision", "recall", "F1", "TP", "FP", "FN"):
            t.add_column(col, justify="right")
        _prf_row(t, "merges", report.resolution)
        console.print(t)
        console.print("[dim]TP=correct merges  FP=over-merges (bad)  FN=missed duplicates[/dim]")


@app.command("eval-compare")
def eval_compare(
    extract_models: str = typer.Option(None, "--extract-models", help="Comma-separated extract models to A/B (e.g. 'llama3.1:8b,qwen2.5:7b')."),
    embed_models: str = typer.Option(None, "--embed-models", help="Comma-separated EMBEDDING models to A/B (e.g. 'nomic-embed-text,mxbai-embed-large'). Dimensions auto-detected."),
    gold: str = typer.Option("eval/gold.json", "--gold", help="Gold-set path."),
    k: int = typer.Option(None, "--k", help="Retrieval cutoff k."),
):
    """A/B several models on the same gold set and print a delta table.

    Vary EITHER --extract-models (moves the graph: ext/triple/res F1) OR --embed-models
    (moves retrieval: precision@R / MAP / nDCG). Each variant re-ingests the gold corpus
    into an isolated in-memory brain, which is dimension-agnostic, so embedding models of
    different dimensions can be compared with no Neo4j index changes.
    """
    from secbrn.eval import Evaluator, load_goldset

    if bool(extract_models) == bool(embed_models):
        console.print("[red]Pass exactly one of --extract-models or --embed-models.[/red]")
        raise typer.Exit(2)

    gs = load_goldset(gold)
    base = get_settings()
    cp = gs.corpus_path()

    if embed_models:
        varying, values, title = "embed_model", [m.strip() for m in embed_models.split(",") if m.strip()], "Model A/B (embedding model)"
    else:
        varying, values, title = "extract_model", [m.strip() for m in extract_models.split(",") if m.strip()], "Model A/B (extract model)"
    if len(values) < 2:
        console.print("[yellow]Give at least two models to compare.[/yellow]")

    results = []
    for m in values:
        update = {varying: m}
        if varying == "embed_model":
            update["embed_dim"] = 0  # auto-detect; in-memory store doesn't need a fixed dim
        s2 = base.model_copy(update=update)
        console.print(f"[dim]running {varying}={m}…[/dim]")
        brain = Brain.isolated(s2)
        try:
            if gs.retrieval and cp is not None:
                brain.ingest(cp)
            rep = Evaluator(brain, k=k).evaluate(gs)
        finally:
            brain.close()
        results.append((m, rep))

    t = Table(title=title)
    for col in (varying.replace("_", " "), "ext.F1", "triple.F1", "res.F1", "precision@R", "MAP", "nDCG@k"):
        t.add_column(col, justify="right")
    base_metrics = None
    for m, rep in results:
        ent = rep.entities.f1 if rep.entities else 0.0
        tri = rep.triples.f1 if rep.triples else 0.0
        res = rep.resolution.f1 if rep.resolution else 0.0
        rpr = rep.retrieval.r_precision if rep.retrieval else 0.0
        mp = rep.retrieval.map if rep.retrieval else 0.0
        nd = rep.retrieval.ndcg_at_k if rep.retrieval else 0.0
        cur = (ent, tri, res, rpr, mp, nd)
        if base_metrics is None:
            base_metrics = cur
            cells = [f"{x:.2f}" for x in cur]
        else:
            cells = [f"{x:.2f} ({x-b:+.2f})" for x, b in zip(cur, base_metrics)]
        t.add_row(m, *cells)
    console.print(t)
    hint = ("retrieval columns move with the embedding model; graph columns stay flat"
            if embed_models else
            "graph columns (ext/triple/res) move with the extract model; retrieval stays flat")
    console.print(f"[dim]Deltas vs the first model. {hint}.[/dim]")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
