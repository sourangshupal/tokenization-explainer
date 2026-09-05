"""Rich display helpers for the medical tokenizer notebook."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Union

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(force_jupyter=True, width=110)

CHIP_BG = ("cyan", "magenta", "green", "yellow", "blue", "bright_white")

TRAIN_GUIDE: dict[str, dict[str, str]] = {
    "Medical custom BPE": {
        "file": "data/pubmed_train.jsonl",
        "why": "Disjoint PubMed train split — 45k abstracts, domain merges guaranteed.",
        "mixed": "Do not train this BPE on general English — empagliflozin will not compress.",
        "style": "green",
    },
    "General BPE": {
        "file": "data/general_train.jsonl",
        "why": "Wikitext-103 train split — same-size control, proves domain matters.",
        "mixed": "Do not mix with medical text — that defeats the comparison.",
        "style": "blue",
    },
}

Renderable = Union[str, int, float, Text, None]


def _chip_style(piece: str, index: int) -> str:
    """Rich style for one token chip."""
    bg = CHIP_BG[index % len(CHIP_BG)]
    return f"bold black on {bg}"


def token_chips(pieces: Sequence[str]) -> Text:
    """Color each piece as a chip."""
    out = Text()
    if not pieces:
        out.append("(empty)", style="dim")
        return out
    for i, piece in enumerate(pieces):
        if i:
            out.append(" ")
        out.append(f" {piece} ", style=_chip_style(str(piece), i))
    return out


def explain_train_choice(choice: str) -> None:
    """Display training-corpus guidance panel."""
    info = TRAIN_GUIDE.get(choice)
    if info is None:
        fail(f"unknown choice: {choice}")
        return
    body = (
        f"[bold]Use[/]  {info['file']}\n\n"
        f"{info['why']}\n\n"
        f"[yellow]If you mixed them:[/] {info['mixed']}"
    )
    console.print(Panel(body, title=choice, border_style=info["style"]))


def compare_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Renderable]],
    caption: str | None = None,
    title: str | None = None,
) -> Table:
    """Side-by-side comparison table."""
    table = Table(
        title=title,
        caption=caption,
        box=box.SIMPLE_HEAD,
        show_header=True,
        pad_edge=False,
        caption_style="dim italic",
    )
    for i, header in enumerate(headers):
        table.add_column(header, style="bold" if i == 0 else None)
    for row in rows:
        cells: list[str | Text] = []
        for cell in row:
            if cell is None:
                cells.append(Text("—", style="dim"))
            elif isinstance(cell, (int, float)):
                cells.append(str(cell))
            else:
                cells.append(cell)
        table.add_row(*cells)
    console.print(table)
    return table


def show_pieces(pieces: Sequence[str], title: str | None = None) -> None:
    """Print a chip row, optionally inside a titled panel."""
    body = token_chips(pieces)
    if title:
        console.print(Panel(body, title=title, border_style="cyan"))
    else:
        console.print(body)


def side_by_side_segs(
    probe: str,
    encoders: dict[str, Callable[[str], list[str]]],
) -> None:
    """Show token chips for each tokenizer on the same probe string — the visual payoff."""
    console.print(f"\n[bold]Probe:[/] [italic]{probe}[/]")
    for name, encode in encoders.items():
        pieces = encode(probe)
        line = Text()
        line.append(f"  {name:12s}", style="dim")
        line.append(f" ({len(pieces):2d}) ", style="bold")
        line.append_text(token_chips(pieces))
        console.print(line)
    console.print()


def vocab_membership(
    terms: Sequence[str],
    encoders: dict[str, Callable[[str], list[str]]],
) -> None:
    """Show which terms each tokenizer encodes as a single token (vocab membership proof)."""
    table = Table(title="Vocab membership — does the term exist as 1 token?", box=box.SIMPLE_HEAD)
    table.add_column("term", style="bold")
    for name in encoders:
        table.add_column(name, justify="center")
    for term in terms:
        row: list[str] = []
        for encode in encoders.values():
            is_single = len(encode(term)) == 1
            row.append("[green]✓ single[/]" if is_single else f"[red]{len(encode(term))} pieces[/]")
        table.add_row(term, *row)
    console.print(table)


def ascii_fertility_chart(
    data: dict[str, dict[str, float]],
    title: str = "Fertility by tokenizer and domain",
) -> None:
    """ASCII bar chart: rows = tokenizers, columns = domain sets, relative bar width.

    data format: {tokenizer_name: {domain_name: fertility_value}}
    """
    all_values = [v for d in data.values() for v in d.values() if v]
    if not all_values:
        return
    max_val = max(all_values)
    bar_width = 30
    domains = list(next(iter(data.values())).keys())
    domain_styles = ["cyan", "green", "yellow", "magenta"]

    console.print(f"\n[bold]{title}[/]")
    console.print(f"  {'tokenizer':14s}  " + "  ".join(f"{d:>20s}" for d in domains))
    console.print("  " + "─" * (16 + 22 * len(domains)))

    for tok_name, domain_vals in data.items():
        line = Text(f"  {tok_name:14s}  ")
        for i, domain in enumerate(domains):
            val = domain_vals.get(domain, 0.0)
            filled = int((val / max_val) * bar_width) if max_val else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            style = domain_styles[i % len(domain_styles)]
            line.append(f" [{bar}] {val:.3f}", style=style)
        console.print(line)
    console.print()


def ascii_bar(
    values: dict[str, float],
    title: str = "",
    lower_is_better: bool = True,
) -> None:
    """Single-series ASCII bars. Shorter bar = better when lower_is_better."""
    if not values:
        return
    max_val = max(values.values()) or 1.0
    bar_width = 40
    if title:
        console.print(f"\n[bold]{title}[/]")
    if lower_is_better:
        console.print("[dim]shorter bar = fewer tokens = better compression[/]")
    for name, val in values.items():
        filled = int((val / max_val) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        console.print(f"  {name:8s} [{bar}] {val:,.1f}")
    console.print()


def ok(msg: str) -> None:
    """Green success panel for a passed golden assert."""
    console.print(Panel(msg, title="assert ok", border_style="green"))


def fail(msg: str) -> None:
    """Red failure panel."""
    console.print(Panel(msg, title="assert failed", border_style="red"))
