"""Ingest CLI: load -> chunk -> embed -> store.

    python -m rag_revops.ingest --source data/raw --persist data/processed/chroma
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from .chunking import TokenChunker
from .config import load_settings
from .embeddings import CohereEmbedder
from .loaders import load_directory
from .vectorstore import ChromaStore

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    source: Path = typer.Option(Path("data/raw"), help="Directory of raw documents."),
    persist: Path | None = typer.Option(None, help="Override Chroma persist dir."),
    reset: bool = typer.Option(False, help="Drop and rebuild the collection."),
) -> None:
    settings = load_settings()
    if persist is not None:
        settings.vectorstore.persist_dir = str(persist)

    console.print(f"[bold]Loading documents from[/bold] {source}")
    docs = load_directory(source)
    if not docs:
        console.print("[red]No supported documents found.[/red]")
        raise typer.Exit(code=1)
    console.print(f"Loaded [green]{len(docs)}[/green] documents.")

    chunker = TokenChunker(settings.chunking)
    chunks = chunker.chunk_documents(docs)
    sizes = [c.n_tokens for c in chunks]
    console.print(
        f"Produced [green]{len(chunks)}[/green] chunks "
        f"(min={min(sizes)}, max={max(sizes)}, "
        f"avg={sum(sizes) // len(sizes)} tokens)."
    )

    embedder = CohereEmbedder(settings.embeddings)
    store = ChromaStore(settings.vectorstore)
    if reset:
        console.print("[yellow]Reset requested — clearing collection.[/yellow]")

    batch = settings.embeddings.batch_size
    for start in track(range(0, len(chunks), batch), description="Embedding + storing"):
        window = chunks[start : start + batch]
        vectors = embedder.embed_documents([c.text for c in window])
        store.add(window, vectors)

    console.print(f"[bold green]Done.[/bold green] Collection now holds {store.count()} chunks.")


if __name__ == "__main__":
    app()
