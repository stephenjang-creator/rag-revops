"""Natural-language question templates for CUAD clause categories.

CUAD's raw questions are clause-extraction prompts ("Highlight the parts related
to 'Cap On Liability'..."), which are fine as annotations but read nothing like
what a rep or deal-desk analyst actually types. For a meaningful faithfulness
eval — and a better portfolio story — we map each clause category to a natural
question. The verified CUAD answer spans stay as ground truth.

Categories not in this map fall back to a generic template, so the builder never
drops a pair just because we didn't hand-write a question for a rare category.
"""

from __future__ import annotations

QUESTION_TEMPLATES: dict[str, str] = {
    "Parties": "Who are the parties to this agreement?",
    "Agreement Date": "What is the date of this agreement?",
    "Effective Date": "What is the effective date of this agreement?",
    "Expiration Date": "When does this agreement expire?",
    "Renewal Term": "What are the renewal terms of this agreement?",
    "Notice Period To Terminate Renewal": (
        "How much notice is required to terminate the renewal?"
    ),
    "Governing Law": "Which state's or country's law governs this agreement?",
    "Termination For Convenience": (
        "Can a party terminate this agreement for convenience, and how?"
    ),
    "Post-Termination Services": (
        "What obligations survive after this agreement is terminated?"
    ),
    "Cap On Liability": "How is liability capped under this agreement?",
    "Uncapped Liability": (
        "Are there any categories of uncapped or unlimited liability?"
    ),
    "Liquidated Damages": "Does this agreement provide for liquidated damages?",
    "Insurance": "What insurance is each party required to carry?",
    "Audit Rights": "What audit rights does this agreement grant?",
    "Warranty Duration": "What is the duration of any warranties?",
    "License Grant": "What license is granted under this agreement?",
    "Non-Transferable License": "Is the license transferable?",
    "Affiliate License-Licensee": (
        "Does the license extend to the licensee's affiliates?"
    ),
    "Irrevocable Or Perpetual License": (
        "Is the license irrevocable or perpetual?"
    ),
    "Ip Ownership Assignment": (
        "Who owns intellectual property created under this agreement?"
    ),
    "Joint Ip Ownership": "Is any intellectual property jointly owned?",
    "Exclusivity": "Does this agreement grant any exclusivity?",
    "Non-Compete": "Does this agreement contain a non-compete restriction?",
    "Competitive Restriction Exception": (
        "Are there exceptions to the competitive restrictions?"
    ),
    "No-Solicit Of Employees": (
        "Does this agreement restrict soliciting the other party's employees?"
    ),
    "No-Solicit Of Customers": (
        "Does this agreement restrict soliciting the other party's customers?"
    ),
    "Anti-Assignment": (
        "Can this agreement be assigned to another party, and under what conditions?"
    ),
    "Change Of Control": (
        "What happens under a change of control of either party?"
    ),
    "Minimum Commitment": (
        "Is there a minimum purchase or volume commitment?"
    ),
    "Volume Restriction": "Are there any volume restrictions?",
    "Price Restrictions": "Are there any restrictions on pricing?",
    "Revenue/Profit Sharing": (
        "How is revenue or profit shared between the parties?"
    ),
    "Rofr/Rofo/Rofn": (
        "Does this agreement grant a right of first refusal, offer, or negotiation?"
    ),
    "Covenant Not To Sue": "Does this agreement include a covenant not to sue?",
    "Third Party Beneficiary": (
        "Are there any third-party beneficiaries to this agreement?"
    ),
    "Document Name": "What type of agreement is this?",
}

_FALLBACK = 'What does this agreement say regarding "{category}"?'


def question_for(category: str) -> str:
    template = QUESTION_TEMPLATES.get(category)
    if template is not None:
        return template
    return _FALLBACK.format(category=category)
