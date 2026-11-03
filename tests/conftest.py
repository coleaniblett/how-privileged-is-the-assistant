"""The test suite is CPU-only by design (model tests use SmolLM2-135M; subprocess tests already pass CUDA_VISIBLE_DEVICES="").
Hide the GPU from the whole session before torch initialises CUDA, so `uv run pytest` gives the same results with a GPU visible
(tests/test_score_conversations.py loads its model in-process and asserts it is on the CPU). Set TESTS_ALLOW_GPU=1 to opt out."""

import os

if os.environ.get("TESTS_ALLOW_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
