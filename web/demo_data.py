"""The three recorded runs, expressed as event streams in the shared schema.

Lifted from the v2 design prototype's DEMOS. `/api/demo` serves these so a
keyless visitor sees the pipeline work; the client replays them on a timer
through the same renderer the live SSE stream drives.

Per the v2 copy rule these carry NO machine texture — no relevance scores, no
chunk ids, no per-step millisecond timings, and no raw CUAD filenames. Trace
steps read as plain English and contracts read as "Outsourcing agreement — NICE
Ltd." (`ms` is passed empty; the renderer doesn't show it here). The live path
still emits the raw values and the client humanizes them the same way.

Order matters: the first event is a stage and the last is `done`; the router /
rewrite events arrive right after the "Understood" step so the reasoning card
lands before the outcome.
"""

from __future__ import annotations

from . import events as E

_DRAFT_CLAUSE = (
    "All amounts due under this Agreement shall be paid within thirty (30) days "
    "from the date of the invoice. Payments shall be made in full, without "
    "set-off, counterclaim, deduction, or withholding of any kind. Overdue "
    "undisputed amounts shall accrue a late payment charge at the lesser of one "
    "and one-half percent (1½%) per month or the maximum rate permitted by "
    "applicable law."
)


def _draft_run() -> dict:
    return {
        "id": "draft",
        "question": "What's a good way to word payment terms for 30 days net?",
        "events": [
            E.stage("Translated", "Your shorthand became the language contracts actually use", ""),
            E.stage("Understood", "Treated as a request for new drafting language", ""),
            E.route(
                "draft_clause", "Draft clause language",
                "You want reusable wording for an order form — not a lookup inside "
                "one contract.",
            ),
            E.rewrite(
                "What's a good way to word payment terms for 30 days net?",
                "Payment shall be due within thirty (30) days of the invoice date.",
                True,
            ),
            E.stage("Searched", "Looked across the whole contract book, two ways at once", ""),
            E.stage("Narrowed", "Kept the six passages that most literally say this", ""),
            E.stage("Drafted", "Wrote wording from those passages, nothing invented", ""),
            E.answer(
                "draft",
                heading="Suggested wording",
                subheading="Payment Terms",
                body_html=_DRAFT_CLAUSE,
                body_plain=_DRAFT_CLAUSE,
                note=(
                    "the examples all support 30 days from the invoice date, though "
                    'one uses 60. If you\'d rather leave the window open, swap in "[__] '
                    'days" and confirm it with the counterparty.'
                ),
                disclaimer="Adapt before use — this is drafting help, not legal advice.",
                sources=[
                    {"doc_id": "Outsourcing agreement — NICE Ltd.", "score": ""},
                    {"doc_id": "Maintenance agreement — Azul S.A.", "score": ""},
                    {"doc_id": "Outsourcing agreement — Oasys Mobile", "score": ""},
                    {"doc_id": "Supply agreement — Upjohn", "score": ""},
                    {"doc_id": "License agreement — CytoDyn", "score": ""},
                    {"doc_id": "Services agreement — Ability Inc.", "score": ""},
                ],
            ),
            E.done(),
        ],
    }


def _find_run() -> dict:
    findings = [
        {"doc_id": "Outsourcing agreement — NICE Ltd.", "tag": "without cause",
         "reason": "Either party may terminate without cause on ninety days’ written notice."},
        {"doc_id": "Services agreement — Cano Health", "tag": "sole discretion",
         "reason": 'The client may end the engagement "in its sole discretion" with '
                   "thirty days’ notice and no penalty."},
        {"doc_id": "Reseller agreement — Aeon Global", "tag": "for any reason",
         "reason": 'Termination "for any reason or no reason" on notice — invisible to '
                   "a keyword search."},
        {"doc_id": "Distribution agreement — Sonic Foundry", "tag": "without penalty",
         "reason": 'Either party may withdraw "without penalty or further obligation" '
                   "after the initial term."},
    ]
    return {
        "id": "find",
        "question": "Which contracts allow termination for convenience?",
        "events": [
            E.stage("Translated", "Widened to cover the other ways contracts say this", ""),
            E.stage("Understood", "Treated as a search across every contract", ""),
            E.route(
                "find_contracts", "Find contracts with clauses",
                "This asks about the whole book, not about one agreement.",
            ),
            E.rewrite(
                "Which contracts allow termination for convenience?",
                "Agreements that let a party end the contract without cause.",
                True,
            ),
            E.stage("Gathered", "Pulled in all one hundred contracts — nothing capped", ""),
            E.stage("Ordered", "Ranked them for reading order only, not for the answer", ""),
            *[E.judge(f["doc_id"], True, f["tag"], f["reason"]) for f in findings],
            E.stage("Read", "Judged each one on meaning; sixteen qualified", ""),
            E.answer(
                "find",
                summary="Sixteen of your hundred contracts qualify",
                findings=findings,
                footer='Not one of these says "termination for convenience." Searching '
                       "for the words would have missed all of them; reading for meaning "
                       "finds them.",
            ),
            E.done(),
        ],
    }


def _decline_run() -> dict:
    return {
        "id": "decline",
        "question": "What does the Pizza Fusion agreement say about termination?",
        "events": [
            E.stage("Translated", "Restated in the language the contract would use", ""),
            E.stage("Understood", "Treated as a question about one named agreement", ""),
            E.route(
                "ask_one_contract", "Ask about a contract",
                "A named agreement and a specific clause inside it.",
            ),
            E.rewrite(
                "What does the Pizza Fusion agreement say about termination?",
                "How the Pizza Fusion franchise agreement can be ended.",
                True,
            ),
            E.stage("Found it", "Matched the agreement you meant", ""),
            E.stage("Searched", "Looked only inside that document", ""),
            E.stage("Checked", "Nothing it found clearly answers the question", ""),
            E.decline(
                reason="I couldn't find support for that in these documents.",
                why="Nothing it retrieved clearly answered the question, so it offered "
                    "nothing. This is the behaviour the whole thing is built around — "
                    "for a deal desk, a confident wrong answer costs far more than a "
                    "missing one.",
            ),
            E.done(),
        ],
    }


def demo_runs() -> list[dict]:
    return [_draft_run(), _find_run(), _decline_run()]
