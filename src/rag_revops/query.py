"""Query CLI: ask a question, get a cited answer (or an honest decline).

    python -m rag_revops.query "Can a customer terminate for convenience?"
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .graph import RagPipeline

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    question: str = typer.Argument(..., help="Your question."),
) -> None:
    pipeline = RagPipeline()
    result = pipeline.ask(question)

    style = "yellow" if result.declined else "green"
    console.print(Panel(result.answer, title="Answer", border_style=style))

    if result.citations:
        console.print("\n[bold]Sources[/bold]")
        for c in result.citations:
            console.print(
                f"  [cyan][{c.marker}][/cyan] {c.source_path} "
                f"[dim](chunk {c.chunk_id}, score {c.score:.3f})[/dim]"
            )
            console.print(f"      [dim]{c.snippet}…[/dim]")
    elif not result.declined:
        console.print("[red]Warning: answer produced no inline citations.[/red]")


if __name__ == "__main__":
    app()
