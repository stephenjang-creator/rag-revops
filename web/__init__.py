"""Render web service for Deal Desk Helper.

A FastAPI app that (a) streams the existing `rag_revops` pipeline as Server-Sent
Events so the page can show the pipeline working *as it happens*, and (b) serves
the static case-study page. One event schema, two sources: the recorded runs
(`/api/demo`) and the live SSE (`/api/ask`) render through the same trace panel,
so a keyless visitor still watches the pipeline work.
"""
