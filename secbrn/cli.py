"""`secbrn` command-line interface — a thin wrapper over :class:`Brain`.

Commands: ingest, ask, search, stats, resolve, reindex, reset, healthcheck,
eval, eval-answers, eval-compare, eval-sweep.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
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
    concurrency: int = typer.Option(None, "--concurrency", "-j", help="Parallel embed/extract calls during ingest (default: SECBRN_INGEST_CONCURRENCY)."),
    progress: bool = typer.Option(None, "--progress/--no-progress", help="Show a live progress bar (default: on when attached to a terminal)."),
):
    """Ingest a file/folder or a URL into the brain (Stages 1–6)."""
    if not path and not url:
        console.print("[red]Provide a PATH or --url[/red]")
        raise typer.Exit(2)
    _banner()
    brain = _brain()
    if concurrency is not None:
        brain.s.ingest_concurrency = max(1, concurrency)

    show_progress = progress if progress is not None else sys.stderr.isatty()

    def _run(cb):
        if url:
            return brain.ingest_url(url, resolve=not no_resolve, progress=cb)
        return brain.ingest(path, resolve=not no_resolve, raise_errors=debug, progress=cb)

    try:
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as prog:
                tasks: dict[tuple[str, str], int] = {}

                def cb(ev):
                    key = (ev.doc_title, ev.phase)
                    if key not in tasks:
                        label = f"[cyan]{ev.phase}[/cyan] {ev.doc_title[:32]}"
                        tasks[key] = prog.add_task(label, total=max(ev.total, 1))
                    prog.update(tasks[key], completed=ev.done, total=max(ev.total, 1))

                rep = _run(cb)
        else:
            rep = _run(None)
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
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Delete ALL data from the graph (documents, chunks, entities, relations).

    Indexes/constraints are kept, so you can immediately re-ingest. Use this when
    idempotent dedup keeps skipping files as 'unchanged' and you want a clean reload.
    """
    s = get_settings()
    backend = resolve_backend(s)
    _banner()
    if backend == "memory":
        console.print(
            "[yellow]Nothing to reset:[/yellow] the in-memory backend isn't persisted, "
            "so each process already starts empty."
        )
        raise typer.Exit(0)

    brain = _brain()
    try:
        before = brain.stats()
        n = before.get("documents", 0) + before.get("chunks", 0) + before.get("entities", 0)
        if n == 0:
            console.print("[green]Graph is already empty.[/green] Nothing to delete.")
            raise typer.Exit(0)
        if not yes:
            console.print(
                f"[red]About to permanently delete ALL data[/red] from neo4j @ {s.neo4j_uri} "
                f"(db={s.neo4j_database}):\n"
                f"  documents={before.get('documents', 0)}  chunks={before.get('chunks', 0)}  "
                f"entities={before.get('entities', 0)}  relations={before.get('relations', 0)}"
            )
            if not typer.confirm("Proceed?"):
                console.print("Aborted — nothing was deleted.")
                raise typer.Exit(1)
        brain.store.clear()
        after = brain.stats()
    finally:
        brain.close()
    console.print(
        f"[green]Graph cleared.[/green] documents={after.get('documents', 0)} "
        f"chunks={after.get('chunks', 0)} entities={after.get('entities', 0)}"
    )
    console.print("Now re-ingest, e.g. [bold]secbrn ingest <path>[/bold].")


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


@app.command("eval-answers")
def eval_answers(
    questions: str = typer.Option("eval/answers.json", "--questions", "-q", help="Gold answers JSON (see eval/answers.example.json)."),
    k: int = typer.Option(None, "--k", help="Chunks to retrieve per question."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the LLM-as-judge; report lexical signals only."),
    judge_model: str = typer.Option(None, "--judge-model", help="Model to grade answers (default: the answer model)."),
    corpus: str = typer.Option(None, "--corpus", help="Ingest this folder into an isolated in-memory brain instead of using the live store (reproducible)."),
    show_cases: bool = typer.Option(False, "--show-cases", help="Print per-question detail."),
):
    """Score generated ANSWERS against gold answers (LLM-judge + key-fact recall).

    By default this asks your LIVE store, so you measure answer quality on the data you
    actually ingested. Pass --corpus to grade against a fresh, isolated ingest instead.
    """
    from pathlib import Path
    from secbrn.eval import AnswerEvaluator, load_answer_set
    from secbrn.providers import get_answer_llm

    if not Path(questions).exists():
        console.print(
            f"[red]No questions file at {questions}.[/red] Copy the template:\n"
            "  [bold]cp eval/answers.example.json eval/answers.json[/bold]  then edit it, "
            "or pass [bold]--questions <path>[/bold]."
        )
        raise typer.Exit(2)
    cases = load_answer_set(questions)
    _banner()

    if corpus:
        brain = Brain.isolated()
        rep = brain.ingest(corpus)
        console.print(f"[dim]isolated corpus {corpus} (chunks={rep.chunks_written}, merges={rep.merges})[/dim]")
    else:
        brain = _brain()

    judge = None
    if not no_judge:
        if judge_model:
            judge = get_answer_llm(get_settings().model_copy(update={"answer_model": judge_model}))
        else:
            judge = brain.answer_llm

    try:
        report = AnswerEvaluator(brain, judge=judge, k=k).evaluate(cases)
    finally:
        brain.close()

    t = Table(title=f"Answer quality (n={report.n})")
    for col in ("correct /5", "complete /5", "key-fact recall", "lexical F1", "grounded"):
        t.add_column(col, justify="right")
    t.add_row(
        f"{report.judge_correct:.2f}", f"{report.judge_complete:.2f}",
        f"{report.key_fact_recall:.2f}", f"{report.lexical_f1:.2f}",
        f"{report.grounded_rate*100:.0f}%",
    )
    console.print(t)
    if no_judge or not report.judged_by_llm:
        console.print(
            "[yellow]LLM judge not used[/yellow] — correct/complete are derived from lexical "
            "overlap with your reference answers. Key-fact recall + grounded are exact."
            if no_judge else
            "[yellow]Judge produced no parseable scores[/yellow] (model didn't return JSON); "
            "correct/complete fell back to lexical overlap."
        )
    else:
        console.print("[dim]correct/complete graded by the LLM judge vs your reference answers.[/dim]")
    console.print("[dim]grounded = answer carried inline citations.[/dim]")

    if show_cases:
        for r in report.per_case:
            mark = "[green]●[/green]" if r.judge_correct >= 3 else "[red]●[/red]"
            console.print(
                f"  {mark} correct={r.judge_correct:.1f} complete={r.judge_complete:.1f} "
                f"kfr={r.key_fact_recall:.2f} f1={r.lexical_f1:.2f} cites={r.n_citations}  "
                f"[dim]{r.query}[/dim]"
            )
            if r.judge_reason:
                console.print(f"      [dim]{r.judge_reason}[/dim]")


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


@app.command("eval-sweep")
def eval_sweep(
    gold: str = typer.Option("eval/gold.json", "--gold"),
    graph_boost: str = typer.Option(None, "--graph-boost", help="Comma list, e.g. '0,0.5,1.0'."),
    title_boost: str = typer.Option(None, "--title-boost", help="Comma list, e.g. '0,0.4,0.8'."),
    top_k: str = typer.Option(None, "--top-k", help="Comma list, e.g. '6,10'."),
    metric: str = typer.Option("map", "--metric", help="Sort by: map | ndcg | r_precision."),
    top: int = typer.Option(10, "--top", help="Rows to show."),
):
    """Grid-search retrieval params against the gold set and report the best config.

    Ingests the corpus once, then sweeps graph_boost / title_boost / top_k at query time
    (no re-ingest). Free tuning — copy the winning row into your .env.
    """
    from itertools import product
    from secbrn.eval import Evaluator, load_goldset

    base = get_settings()

    def _vals(opt, default):
        if not opt:
            return [default]
        out = []
        for x in opt.split(","):
            x = x.strip()
            if x:
                out.append(type(default)(x))
        return out or [default]

    gbs = _vals(graph_boost, base.graph_boost)
    tbs = _vals(title_boost, base.title_boost)
    tks = _vals(top_k, base.retrieve_top_k)
    metric = metric.lower()
    if metric not in ("map", "ndcg", "r_precision"):
        console.print("[red]--metric must be map | ndcg | r_precision[/red]")
        raise typer.Exit(2)

    gs = load_goldset(gold)
    brain = Brain.isolated(base)
    rows = []
    try:
        if gs.retrieval and gs.corpus_path() is not None:
            rep = brain.ingest(gs.corpus_path())
            console.print(f"[dim]ingested corpus (chunks={rep.chunks_written}); sweeping "
                          f"{len(gbs)*len(tbs)*len(tks)} combos…[/dim]")
        for gb, tb, tk in product(gbs, tbs, tks):
            brain.s.graph_boost = gb
            brain.s.title_boost = tb
            brain.s.retrieve_top_k = tk
            r = Evaluator(brain, k=tk).evaluate(gs).retrieval
            rows.append((gb, tb, tk, r.r_precision, r.map, r.ndcg_at_k))
    finally:
        brain.close()

    key = {"map": 4, "ndcg": 5, "r_precision": 3}[metric]
    rows.sort(key=lambda t: t[key], reverse=True)

    t = Table(title=f"Retrieval sweep (sorted by {metric})")
    for col in ("graph_boost", "title_boost", "top_k", "precision@R", "MAP", "nDCG@k"):
        t.add_column(col, justify="right")
    for gb, tb, tk, rp, mp, nd in rows[:top]:
        t.add_row(f"{gb:g}", f"{tb:g}", str(tk), f"{rp:.3f}", f"{mp:.3f}", f"{nd:.3f}")
    console.print(t)
    if rows:
        gb, tb, tk, rp, mp, nd = rows[0]
        console.print(f"[green]Best:[/green] SECBRN_GRAPH_BOOST={gb:g} SECBRN_TITLE_BOOST={tb:g} "
                      f"SECBRN_RETRIEVE_TOP_K={tk}  ({metric}={rows[0][key]:.3f})")


def main():  # pragma: no cover
    """Console-script entry point (see pyproject [project.scripts])."""
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
