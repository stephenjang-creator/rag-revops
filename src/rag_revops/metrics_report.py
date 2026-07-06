"""Phase 2 — SRE metrics over the request sink, and Phase 3 — gate on them.

Reads the JSONL written by ``observability.RequestTimer`` and computes the
site-reliability signals that averages hide:

  * latency p50 / p95 / p99   — tail latency, not just the mean
  * llm_calls per request     — p50 / max (cost & fan-out)
  * citation_coverage         — % of *answered* (non-declined, non-error)
                                requests that carry >=1 inline citation
  * failure_rate              — % of requests that errored
  * decline_rate              — % that declined (context, not a failure)

Usage::

    # human-readable report over the whole sink
    python -m rag_revops.metrics_report

    # only the last N minutes, JSON out
    python -m rag_revops.metrics_report --since-min 60 --json

    # CI gate: exit 1 if coverage/failure breach thresholds
    python -m rag_revops.metrics_report --gate \
        --min-coverage 0.90 --max-failure-rate 0.02 --max-p95-ms 60000

The gate is deliberately separate from the faithfulness gate (eval/run_eval.py):
faithfulness needs live API calls and golden data; these operational metrics
come for free from traffic the pipeline already served, so they can gate on
every run without extra API spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. Small-sample honest: no interpolation games."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    # nearest-rank: rank = ceil(pct/100 * N), clamped to [1, N]
    import math

    rank = max(1, min(len(s), math.ceil(pct / 100.0 * len(s))))
    return s[rank - 1]


def load_records(path: Path, since_min: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = None
    if since_min is not None:
        cutoff = datetime.now(UTC) - timedelta(minutes=since_min)
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cutoff is not None:
            ts = rec.get("ts")
            if ts:
                try:
                    when = datetime.fromisoformat(ts)
                    if when < cutoff:
                        continue
                except ValueError:
                    pass
        out.append(rec)
    return out


def compute(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    latencies = [r["latency_ms"] for r in records if isinstance(r.get("latency_ms"), (int, float))]
    calls = [r["llm_calls"] for r in records if isinstance(r.get("llm_calls"), int)]

    errors = [r for r in records if r.get("error")]
    declines = [r for r in records if r.get("declined")]
    # Coverage denominator: answered requests only (exclude declines and errors).
    grounded_flags = [r["grounded"] for r in records if r.get("grounded") is not None]
    n_grounded_true = sum(1 for g in grounded_flags if g)

    def rate(numer: int, denom: int) -> float | None:
        return (numer / denom) if denom else None

    return {
        "n_requests": n,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "max": max(latencies) if latencies else None,
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
        },
        "llm_calls": {
            "p50": _percentile([float(c) for c in calls], 50),
            "max": max(calls) if calls else None,
            "total": sum(calls) if calls else 0,
        },
        "citation_coverage": rate(n_grounded_true, len(grounded_flags)),
        "n_answered_scored_for_coverage": len(grounded_flags),
        "failure_rate": rate(len(errors), n),
        "n_errors": len(errors),
        "decline_rate": rate(len(declines), n),
        "n_declines": len(declines),
    }


def _fmt(v: float | None) -> str:
    """Latency-style: one decimal (milliseconds)."""
    return "n/a" if v is None else f"{v:.1f}"


def _rate(v: float | None) -> str:
    """Rate-style: three decimals + percent, so small rates aren't hidden."""
    return "n/a" if v is None else f"{v:.3f} ({v * 100:.1f}%)"


def print_report(m: dict[str, Any]) -> None:
    lat = m["latency_ms"]
    print("=" * 52)
    print("OPERATIONAL METRICS")
    print("=" * 52)
    print(f"Requests observed:     {m['n_requests']}")
    print("-" * 52)
    print("Latency (ms)")
    print(f"  p50 {_fmt(lat['p50'])}   p95 {_fmt(lat['p95'])}   "
          f"p99 {_fmt(lat['p99'])}   max {_fmt(lat['max'])}")
    print(f"  mean {_fmt(lat['mean'])}   (mean hides the tail — watch p95)")
    print("-" * 52)
    print("LLM calls per request")
    print(f"  p50 {m['llm_calls']['p50']}   max {m['llm_calls']['max']}   "
          f"total {m['llm_calls']['total']}")
    print("-" * 52)
    print(f"Citation coverage:     {_rate(m['citation_coverage'])}"
          f"   ({m['n_answered_scored_for_coverage']} answered requests)")
    print(f"Failure rate:          {_rate(m['failure_rate'])}"
          f"   ({m['n_errors']} errors)")
    print(f"Decline rate:          {_rate(m['decline_rate'])}"
          f"   ({m['n_declines']} declines)")
    print("=" * 52)


def apply_gate(
    m: dict[str, Any],
    *,
    min_coverage: float | None,
    max_failure_rate: float | None,
    max_p95_ms: float | None,
) -> int:
    """Return 0 if all provided thresholds pass, 1 otherwise. A threshold whose
    metric is unmeasurable (no data) is skipped with a warning rather than
    failing the build on absence of traffic."""
    failed = False

    if min_coverage is not None:
        cov = m["citation_coverage"]
        if cov is None:
            print("::warning::citation_coverage not measurable (no answered requests).")
        elif cov < min_coverage:
            print(f"❌ coverage {cov:.3f} < min {min_coverage:.3f}", file=sys.stderr)
            failed = True
        else:
            print(f"✅ coverage {cov:.3f} >= {min_coverage:.3f}")

    if max_failure_rate is not None:
        fr = m["failure_rate"]
        if fr is None:
            print("::warning::failure_rate not measurable (no requests).")
        elif fr > max_failure_rate:
            print(f"❌ failure_rate {fr:.3f} > max {max_failure_rate:.3f}", file=sys.stderr)
            failed = True
        else:
            print(f"✅ failure_rate {fr:.3f} <= {max_failure_rate:.3f}")

    if max_p95_ms is not None:
        p95 = m["latency_ms"]["p95"]
        if p95 is None:
            print("::warning::p95 latency not measurable (no requests).")
        elif p95 > max_p95_ms:
            print(f"❌ p95 {p95:.0f}ms > max {max_p95_ms:.0f}ms", file=sys.stderr)
            failed = True
        else:
            print(f"✅ p95 {p95:.0f}ms <= {max_p95_ms:.0f}ms")

    return 1 if failed else 0


def main() -> None:
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    default_path = os.environ.get("RAG_METRICS_PATH", "metrics/requests.jsonl")
    ap.add_argument("--path", type=Path, default=Path(default_path))
    ap.add_argument("--since-min", type=int, default=None,
                    help="Only consider requests from the last N minutes.")
    ap.add_argument("--json", action="store_true", help="Emit metrics as JSON.")
    ap.add_argument("--gate", action="store_true", help="Apply thresholds; exit 1 on breach.")
    ap.add_argument("--min-coverage", type=float, default=None)
    ap.add_argument("--max-failure-rate", type=float, default=None)
    ap.add_argument("--max-p95-ms", type=float, default=None)
    args = ap.parse_args()

    records = load_records(args.path, args.since_min)
    metrics = compute(records)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(metrics)

    if args.gate:
        code = apply_gate(
            metrics,
            min_coverage=args.min_coverage,
            max_failure_rate=args.max_failure_rate,
            max_p95_ms=args.max_p95_ms,
        )
        sys.exit(code)


if __name__ == "__main__":
    main()
