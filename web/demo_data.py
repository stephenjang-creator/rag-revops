"""The three recorded runs, expressed as event streams in the shared schema.

Lifted from the design prototype's DEMOS. `/api/demo` serves these so a keyless
visitor sees the pipeline work; the client replays them on a timer through the
same renderer the live SSE stream drives. Order matters: stage rows pace out, and
the router/rewrite cards appear right after the "Route" stage, results at the end.
"""

from __future__ import annotations

from . import events as E

_DRAFT_CLAUSE_HTML = (
    "All amounts due under this Agreement shall be paid within thirty (30) days "
    'from the date of the invoice.<sup>[1][2][3]</sup> Payments shall be made in '
    "full, without set-off, counterclaim, deduction, or withholding of any "
    'kind.<sup>[2]</sup> Overdue undisputed amounts shall accrue a late payment '
    "charge at the lesser of one and one-half percent (1½%) per month or the "
    "maximum rate permitted by applicable law.<sup>[3]</sup>"
)

_DRAFT_PLAIN = (
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
            E.stage("Translate", "Shorthand expanded into contract language", "0.9s"),
            E.stage("Route", "Sent to: draft clause language", "1.4s"),
            E.route(
                "draft_clause", "Draft clause language",
                "Draft clause language — you want reusable wording for an order "
                "form, not a lookup in one contract.",
            ),
            E.rewrite(
                "What's a good way to word payment terms for 30 days net?",
                "Payment shall be due within thirty (30) days of the invoice date.",
                True,
            ),
            E.stage("Retrieve", "412 passages fetched, exact-term + vector, fused on rank", "3.1s"),
            E.stage("Rerank", "Top 6 passages scored 0.875 – 0.999 for precision", "4.0s"),
            E.stage("Draft", "Clause written from those passages, every sentence cited", "9.6s"),
            E.answer(
                "draft",
                heading="Suggested clause language",
                subheading="Payment Terms",
                body_html=_DRAFT_CLAUSE_HTML,
                body_plain=_DRAFT_PLAIN,
                note=(
                    "the examples support a 30-day period from invoice date; one "
                    'uses 60 days. If a longer window is preferred, swap in "[__] '
                    'days" and confirm with the counterparty.'
                ),
                disclaimer=(
                    "Drafted from real contract language and grounded in the "
                    "sources below — adapt before use, not legal advice."
                ),
                sources=[
                    {"doc_id": "041__NICELTD_2003-EX-4.5-OUTSOURCING_AGREEMENT", "score": "0.999"},
                    {"doc_id": "086__AzulSa_2017-EX-10.3-MAINTENANCE_AGREEMENT", "score": "0.998"},
                    {"doc_id": "069__OASYSMOBILE_2001-EX-10.17-OUTSOURCING", "score": "0.993"},
                    {"doc_id": "071__UpjohnInc_2020-EX-2.6-SUPPLY_AGREEMENT", "score": "0.985"},
                    {"doc_id": "039__CytodynInc_2020-EX-10.5-LICENSE_AGREEMENT", "score": "0.967"},
                    {"doc_id": "075__ABILITYINC_2020-EX-4.25-SERVICES_AGREEMENT", "score": "0.875"},
                ],
            ),
            E.done(total_ms=9600, llm_calls=3),
        ],
    }


def _find_run() -> dict:
    findings = [
        {"doc_id": "041__NICELTD-EX-4.5-OUTSOURCING_AGREEMENT", "tag": "without cause",
         "reason": "Either party may terminate without cause on 90 days’ written "
                   "notice — convenience by another name."},
        {"doc_id": "012__CanoHealth-EX-10.9-SERVICES_AGREEMENT", "tag": "sole discretion",
         "reason": 'Client may end the engagement "in its sole discretion" with 30 '
                   "days’ notice and no penalty."},
        {"doc_id": "033__Aeon_Global-EX-10.1-RESELLER_AGREEMENT", "tag": "for any reason",
         "reason": 'Termination "for any reason or no reason" on notice; the judge '
                   "matched it, the reranker scored it 0.00."},
        {"doc_id": "057__SonicFoundry-EX-10.4-DISTRIBUTION", "tag": "without penalty",
         "reason": 'Either party may withdraw "without penalty or further '
                   'obligation" after the initial term.'},
    ]
    return {
        "id": "find",
        "question": "Which contracts allow termination for convenience?",
        "events": [
            E.stage("Translate", "Question widened to cover equivalent wording", "1.1s"),
            E.stage("Route", "Sent to: find contracts with clauses", "1.6s"),
            E.route(
                "find_contracts", "Find contracts with clauses",
                "Find contracts with clauses — this asks across the whole book, "
                "not about one agreement.",
            ),
            E.rewrite(
                "Which contracts allow termination for convenience?",
                "Which agreements include a clause permitting termination without cause?",
                True,
            ),
            E.stage("Retrieve", "Every contract in the corpus pulled in — no result cap", "6.4s"),
            E.stage("Rerank",
                    "Used for reading order only; paraphrased clauses score ~0.00", "11.2s"),
            *[E.judge(f["doc_id"], True, f["tag"], f["reason"]) for f in findings],
            E.stage("Judge", "100 contracts read for legal meaning, 16 confirmed", "38.5s"),
            E.answer("find", summary="16 of 100 contracts qualify", findings=findings,
                     footer="Wording varies wildly — \"without cause\", \"in its sole "
                            "discretion\", \"for any reason\". A keyword search finds none "
                            "of these; a judge that reads for meaning finds all of them."),
            E.done(total_ms=38500, llm_calls=11),
        ],
    }


def _decline_run() -> dict:
    return {
        "id": "decline",
        "question": "What does the Pizza Fusion agreement say about termination?",
        "events": [
            E.stage("Translate", "Question restated in contract language", "0.8s"),
            E.stage("Route", "Sent to: ask about a contract", "1.3s"),
            E.route(
                "ask_one_contract", "Ask about a contract",
                "Ask about a contract — a named agreement and a specific clause within it.",
            ),
            E.rewrite(
                "What does the Pizza Fusion agreement say about termination?",
                "What are the terms and conditions for ending the Pizza Fusion "
                "franchise contract?",
                True,
            ),
            E.stage("Resolve", "Matched the named agreement in the corpus", "1.7s"),
            E.stage("Retrieve", "Passages pulled from that contract only", "3.4s"),
            E.stage("Check", "No passage clearly supports an answer", "6.2s"),
            E.decline(
                reason="Declined — I couldn't find support for that in the documents.",
                why="The retrieved passages didn't clearly answer the question, so "
                    "nothing is offered. This is the behaviour the whole system is "
                    "built around: a wrong answer costs a deal desk far more than a "
                    "missing one.",
            ),
            E.done(total_ms=6200, llm_calls=2),
        ],
    }


def demo_runs() -> list[dict]:
    return [_draft_run(), _find_run(), _decline_run()]
