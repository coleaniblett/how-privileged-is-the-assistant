"""Models for the Shiller user-turn exploration, and their verification through the HF API.

    python -m exploration.shiller_user_turns.models                # verify all -> models.md here
    python -m exploration.shiller_user_turns.models --models small

This exploration is not a measure and its model list is deliberately local: core/config.py still defines
the models of the measures (AGENTS.md). The OLMo IDs are read from config.FAMILIES so they cannot drift.
Base checkpoints are rendered with their family's instruct tokenizer and chat template, as in core.

Verification downloads only config/tokenizer files (no weights) and checks, per model: the repo exists
(sha, license, gating, weight size, architecture), the template repo's chat template places a user
message verbatim, the weights repo's vocabulary matches the template tokenizer's (so the base model
understands the rendered ids), and derives the turn markers and stop token used for generation and flags.
"""

import argparse
import hashlib
import json

from huggingface_hub import HfApi, hf_hub_download

from core import config
from core.generate import stop_ids
from core.render import load_tokenizer
from exploration.shiller_user_turns import render

_OLMO = config.FAMILIES["olmo3-7b"]

# key -> repo (weights), template (tokenizer + chat template), role, large (needs ~65 GB in bf16),
# template_kwargs (passed to apply_chat_template; {} = the template's defaults).
MODELS = {
    "olmo3-7b-instruct": dict(repo=_OLMO["stages"]["instruct"], template=_OLMO["template"], role="instruct"),
    "olmo3-7b-base": dict(repo=_OLMO["stages"]["base"], template=_OLMO["template"], role="base"),
    "qwen3-8b": dict(repo="Qwen/Qwen3-8B", template="Qwen/Qwen3-8B", role="instruct"),
    "qwen3-8b-base": dict(repo="Qwen/Qwen3-8B-Base", template="Qwen/Qwen3-8B", role="base"),
    "gemma4-e4b-it": dict(repo="google/gemma-4-E4B-it", template="google/gemma-4-E4B-it", role="instruct"),
    "gemma4-e4b-base": dict(repo="google/gemma-4-E4B", template="google/gemma-4-E4B-it", role="base"),
    "gemma4-31b-it": dict(repo="google/gemma-4-31B-it", template="google/gemma-4-31B-it", role="instruct", large=True),
    "gemma4-31b-base": dict(repo="google/gemma-4-31B", template="google/gemma-4-31B-it", role="base", large=True),
    # Qwen published no base checkpoint for Qwen3-32B.
    "qwen3-32b": dict(repo="Qwen/Qwen3-32B", template="Qwen/Qwen3-32B", role="instruct", large=True),
}
for _m in MODELS.values():
    _m.setdefault("large", False)
    _m.setdefault("template_kwargs", {})

GROUPS = {
    "small": [k for k, m in MODELS.items() if not m["large"]],
    "large": [k for k, m in MODELS.items() if m["large"]],
    "all": list(MODELS),
}

# Template options whose effect on the rendered prompt is recorded (not changed) for every model.
PROBED_KWARGS = [{"enable_thinking": True}, {"enable_thinking": False}]


def resolve_models(spec):
    """'small' | 'large' | 'all' | comma/space-separated keys (groups may be mixed in)."""
    keys = []
    for part in spec.replace(",", " ").split():
        for k in GROUPS.get(part, [part]):
            if k not in MODELS:
                raise SystemExit(f"unknown model {k!r}; known: {', '.join(MODELS)} or a group {list(GROUPS)}")
            if k not in keys:
                keys.append(k)
    return keys


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12] if s else None


def vocab_of(repo):
    """(vocab incl. added tokens, set of added-token strings) from tokenizer.json; (None, set()) if absent."""
    try:
        t = json.load(open(hf_hub_download(repo, "tokenizer.json")))
    except Exception:  # noqa: BLE001 - no tokenizer.json in the weights repo
        return None, set()
    vocab = dict(t["model"]["vocab"]) if isinstance(t["model"].get("vocab"), dict) else {
        p[0]: i for i, p in enumerate(t["model"].get("vocab", []))}
    added = {a["content"]: a["id"] for a in t.get("added_tokens", [])}
    vocab.update(added)
    return vocab, set(added)


def weight_bytes(info):
    return sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))


def verify(key, api=None):
    """Check one model through the HF API (no weights). Raises on a failed check; returns a record."""
    m = MODELS[key]
    api = api or HfApi()
    info = api.model_info(m["repo"], files_metadata=True)
    tinfo = info if m["template"] == m["repo"] else api.model_info(m["template"])
    cfg = json.load(open(hf_hub_download(m["repo"], "config.json")))
    tok = load_tokenizer(m["template"])
    if not tok.chat_template:
        raise ValueError(f"{m['template']} has no chat template")
    kw = m["template_kwargs"]
    markers = render.turn_markers(tok, kw)  # raises if the template does not place content verbatim
    example = render.render_open_user_turn(tok, "", kw)

    (vocab_t, added_t), (vocab_w, added_w) = vocab_of(m["template"]), vocab_of(m["repo"])
    if vocab_w is None or vocab_w == vocab_t:
        vocab_note = "same" if vocab_w is not None else "weights repo has no tokenizer.json"
    else:
        diff = {k for k in set(vocab_t) | set(vocab_w) if vocab_t.get(k) != vocab_w.get(k)}
        if not diff <= added_t | added_w:  # reserved/added tokens never occur in our prompts (check_text)
            raise ValueError(f"{key}: vocab of {m['repo']} differs from {m['template']} on ordinary tokens: "
                             f"{sorted(diff)[:10]}")
        vocab_note = f"differs only on {len(diff)} added/special tokens"
    text_cfg = cfg.get("text_config", cfg)
    if text_cfg.get("vocab_size", len(tok)) < max(tok.get_vocab().values()) + 1:
        raise ValueError(f"{key}: model vocab_size {text_cfg.get('vocab_size')} < tokenizer size")

    probes = {}
    for p in PROBED_KWARGS:
        try:
            probes[json.dumps(p)] = render.render_open_user_turn(tok, "", {**kw, **p}).text != example.text
        except Exception as e:  # noqa: BLE001 - an option the template rejects is recorded, not fatal
            probes[json.dumps(p)] = f"error: {e}"
    return dict(
        key=key, repo=m["repo"], sha=info.sha, template_repo=m["template"], template_sha=tinfo.sha,
        role=m["role"], large=m["large"], license=(info.card_data or {}).get("license") if info.card_data else None,
        gated=info.gated, weights_gb=round(weight_bytes(info) / 1e9, 1), architectures=cfg.get("architectures"),
        chat_template_sha=sha(tok.chat_template), template_kwargs=kw, kwargs_change_prompt=probes,
        vocab_vs_template=vocab_note, markers=markers,
        stop_ids=stop_ids(tok), stop_tokens=tok.convert_ids_to_tokens(stop_ids(tok)),
        empty_prompt=example.text,
    )


def write_md(records, path):
    lines = ["# Models verified for the Shiller user-turn exploration", "",
             "Written by `python -m exploration.shiller_user_turns.models` (HF API; no weights downloaded).", "",
             "| key | repo | sha | role | license | gated | weights (GB) | architecture | template from | vocab vs template |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['key']} | `{r['repo']}` | {r['sha'][:10]} | {r['role']} | {r['license']} | {r['gated']} | "
                     f"{r['weights_gb']} | {', '.join(r['architectures'] or [])} | `{r['template_repo']}` | "
                     f"{r['vocab_vs_template']} |")
    lines += ["", "## Turn markers, stop token and empty prompt", "",
              "Generation stops only at the stop token (end of text), never at end of turn. "
              "`kwargs change prompt` says whether passing the option would change the rendered open user turn; "
              "the run uses the template defaults (`template_kwargs`).", ""]
    for r in records:
        lines += [f"### {r['key']}", "",
                  f"- markers: `{json.dumps(r['markers'], ensure_ascii=False)}`",
                  f"- stop: {r['stop_tokens']} (ids {r['stop_ids']})",
                  f"- template_kwargs: `{json.dumps(r['template_kwargs'])}`; kwargs change prompt: "
                  f"`{json.dumps(r['kwargs_change_prompt'])}`",
                  f"- chat template sha: {r['chat_template_sha']}", "",
                  "Empty-condition prompt (exact string):", "", "````", r["empty_prompt"], "````", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--out", default="exploration/shiller_user_turns/models.md")
    args = ap.parse_args()
    api, records, failed = HfApi(), [], {}
    for key in resolve_models(args.models):
        try:
            records.append(verify(key, api))
            print(f"ok   {key}: {records[-1]['repo']} @ {records[-1]['sha'][:10]}")
        except Exception as e:  # noqa: BLE001
            failed[key] = repr(e)
            print(f"FAIL {key}: {e!r}")
    write_md(records, args.out)
    print(f"wrote {args.out}")
    if failed:
        raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()
