"""Shared pipeline: config (models, families, stages, slots, paths), model checks, data prep and
contamination filtering, rendering, scoring, sampling, and the loading/statistics helpers every
measure builds on. Nothing in core imports from measures/ or extras/.

Run modules from the repo root with python -m, e.g. `uv run python -m core.score --model sft`.
"""
