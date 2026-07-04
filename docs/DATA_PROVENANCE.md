# Data Provenance & Licensing

This project uses **only public data**. No proprietary, employer, or customer
data is used anywhere in the corpus, the evaluation set, or the examples.

## Sources

### CUAD — Contract Understanding Atticus Dataset
- **What:** 510 real commercial contracts with 13,000+ expert annotations across
  41 clause categories.
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Use here:** Sample contracts form the demo corpus; the clause annotations
  (`master_clauses.csv`) seed the Phase 3 golden evaluation set.
- **Attribution:** Hendrycks, Burns, Chen, Ball. "CUAD: An Expert-Annotated NLP
  Dataset for Legal Contract Review." The Atticus Project.

### Publicly published SaaS agreements & SLAs
- **What:** Model and production SaaS master agreements, service descriptions,
  and SLAs that vendors post publicly on their own sites.
- **License:** Copyrighted by their respective owners.
- **Use here:** Downloaded locally on demand via `scripts/download_corpus.py`.
  **Raw files are not committed to this repo and are not redistributed.** The
  script records the source URL for each. Review each source's terms before any
  redistribution.

## Handling note (why this matters for an ops project)

Data provenance and licensing hygiene is itself part of the deliverable. A RAG
system that can't say where its knowledge came from — or that quietly ingests
material it has no right to redistribute — is a governance liability. This repo
treats "where did this come from and are we allowed to use it" as a first-class
concern, mirroring how a real RevOps/PMO function should evaluate an internal
knowledge assistant before it touches contract or customer data.
