"""Analytical query CLI: ask 'which contracts have X' across the whole corpus.

    python -m rag_revops.query_analytical "Which contracts allow termination for convenience?"
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .analytical import AnalyticalRetriever
from .analytical_generation import AnalyticalGenerator
from .config import load_settings
from .embeddings import CohereEmbedder
from .vectorstore import ChromaStore

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(question: str = typer.Argument(..., help="A 'which contracts have X' question.")) -> None:
    settings = load_settings()
    store = ChromaStore(settings.vectorstore)
    embedder = CohereEmbedder(settings.embeddings)
    retriever = AnalyticalRetriever(settings, store, embedder)
    generator = AnalyticalGenerator(settings)

    with console.status("Searching across the corpus…"):
        matches = retriever.retrieve(question)
        result = generator.generate(question, matches)

    console.print(Panel(result.answer, title="Analytical answer", border_style="green"))
    console.print(
        f"[dim]{result.n_candidates} candidate contract(s) retrieved; "
        f"{len(result.findings)} confirmed.[/dim]"
    )
    if result.findings:
        console.print("\n[bold]Matches[/bold]")
        for f in result.findings:
            console.print(
                f"  [cyan]{f.doc_id}[/cyan] [dim](relevance {f.score:.3f})[/dim]\n"
                f"      {f.reason}"
            )


if __name__ == "__main__":
    app()
