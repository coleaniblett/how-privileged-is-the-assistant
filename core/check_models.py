"""Verify model IDs, licenses and chat templates via the HF API (no weights downloaded).

    python -m core.check_models                     # the default family (OLMo 3 7B): rewrites docs/models.md
    python -m core.check_models --family gemma4-31b # another family: adds/replaces its section in docs/models.md
    python -m core.check_models --family all

Downloads the template checkpoint's tokenizer + config and fetches only the tokenizer files of the
other checkpoints, and checks the post-trained checkpoints share one chat template and all checkpoints
of a family share one tokenizer. Both checks are semantic: templates are compared by what they render
for a plain user/assistant exchange (the files differ in a tool-handling null check), tokenizers by
vocab + merges + added tokens (the files differ in how merges are serialized and in the names/ids of a
few reserved tokens). For every family it also records what core derives from the template: that the
template places content verbatim in each slot, the end-of-turn token, the user->assistant handoff string,
and the assistant-slot opening (core.render), next to the settings in config.FAMILIES.
"""

import hashlib
import json

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from core import config

DEMO = [{"role": "user", "content": "USER"}, {"role": "assistant", "content": "ASSISTANT"}]


def chat_template(repo, files):
    if "chat_template.jinja" in files:
        return open(hf_hub_download(repo, "chat_template.jinja")).read()
    return json.load(open(hf_hub_download(repo, "tokenizer_config.json"))).get("chat_template")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12] if s else None


def tokenizer_core(path):
    """(vocab, merges, added tokens) with merges normalized to pairs regardless of serialization."""
    t = json.load(open(path))
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in t["model"]["merges"]]
    added = {(a["id"], a["content"]) for a in t.get("added_tokens", [])}
    return t["model"]["vocab"], merges, added


def diff_vocab(a, b):
    return sorted((k, a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k))


def as_dict(vocab):
    return dict(vocab) if isinstance(vocab, dict) else {t: i for i, (t, *_) in enumerate(vocab)}


def check_family(name):
    """Markdown lines for one family; asserts the post-trained templates and all tokenizers agree."""
    from core import render
    fam = config.FAMILIES[name]
    stages, template = fam["stages"], fam["template"]
    post = [s for s in stages if s != "base"]
    api = HfApi()
    title = "# Model checks" if name == "olmo3-7b" else f"# Model checks: {name}"
    lines = [title, "", "| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |",
             "|---|---|---|---|---|---|---|---|---|"]
    templates, tok_hashes, cores = {}, {}, {}
    for stage, repo in stages.items():
        info = api.model_info(repo, files_metadata=True)
        files = [s.rfilename for s in info.siblings]
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9
        license_ = (info.card_data or {}).get("license") if info.card_data else None
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        cfg = cfg.get("text_config", cfg) if cfg.get("vocab_size") is None else cfg
        templates[stage] = chat_template(repo, files)
        if "tokenizer.json" in files:
            path = hf_hub_download(repo, "tokenizer.json")
            tok_hashes[stage] = sha(open(path).read())
            cores[stage] = tokenizer_core(path)
        else:
            tok_hashes[stage] = cores[stage] = None
        lines.append(f"| {stage} | `{repo}` | {info.sha[:10]} | {license_} | {info.gated} | {size:.1f} | "
                     f"{cfg.get('vocab_size')} | {sha(templates[stage])} | {tok_hashes[stage]} |")

    tok = AutoTokenizer.from_pretrained(template)
    AutoConfig.from_pretrained(template)
    ref_stage = "instruct"

    # Chat templates: byte equality, then rendering equality on a plain exchange (no tools).
    same_bytes = len({templates[s] for s in post}) == 1
    renders = {s: tok.apply_chat_template(DEMO, chat_template=templates[s], tokenize=False) for s in post}
    same_render = len(set(renders.values())) == 1
    lines += ["", "## Chat templates", "",
              f"Post-trained checkpoints share one chat template file: **{same_bytes}**.",
              f"Post-trained templates render a plain user/assistant exchange identically: **{same_render}**."]
    if not same_bytes:
        import difflib
        for other in sorted({templates[s] for s in post} - {templates[ref_stage]}):
            who = ", ".join(s for s in post if templates[s] == other)
            d = difflib.unified_diff(templates[ref_stage].splitlines(), other.splitlines(), "instruct", who, lineterm="", n=0)
            lines += ["", f"Template file of {who} vs instruct (unified diff):", "", "```diff", *list(d)[2:], "```"]
    if "base" in stages:
        lines.append(f"\nBase has its own chat template: {templates['base'] is not None}; it is scored with the Instruct template.")

    # Tokenizers: byte equality, then vocab/merges/added-token equality.
    ref = cores[ref_stage]
    same_tok_bytes = len(set(tok_hashes.values())) == 1
    same_merges = all(c[1] == ref[1] for c in cores.values())
    vocab_diffs = {s: diff_vocab(as_dict(c[0]), as_dict(ref[0])) for s, c in cores.items()}
    added_diffs = {s: (c[2] - ref[2], ref[2] - c[2]) for s, c in cores.items()}
    same_vocab = all(not d for d in vocab_diffs.values())
    n = len(stages)
    lines += ["", "## Tokenizers", "",
              f"All {n} share one tokenizer.json file: **{same_tok_bytes}**.",
              f"All {n} share one merge list: **{same_merges}**.",
              f"All {n} share one vocab: **{same_vocab}**."]
    for s, d in vocab_diffs.items():
        if d:
            lines += ["", f"`{s}` vocab entries differing from Instruct (token, {s} id, instruct id):", ""]
            lines += [f"- `{k}`: {a} vs {b}" for k, a, b in d]
    # Differences among reserved/added tokens (never present in scored text, see render.check_text)
    # are harmless; a differing id for an ordinary vocab entry is not.
    reserved = {name_ for c in cores.values() for _, name_ in c[2]}
    only_reserved = all(k in reserved for d in vocab_diffs.values() for k, _, _ in d)
    lines.append(f"\nVocab differences are confined to reserved/added tokens: **{only_reserved}**.")

    # What core derives from the template for this family, and the family's config.
    st = config.family_settings(name)
    try:
        parts = {slot: render.template_parts(tok, slot) for slot in ("user", "assistant")}
        verbatim = True
    except Exception as e:  # noqa: BLE001
        parts, verbatim = {}, f"no ({e})"
    end_of_turn, handoff = render.turn_markers(tok)
    lines += ["", "## Derived by core.render from the template", "",
              f"Template places content verbatim exactly once in each slot: **{verbatim}**.", "",
              f"- user slot (prefix | suffix): `{parts.get('user')}`", f"- assistant slot (prefix | suffix): `{parts.get('assistant')}`",
              f"- end-of-turn token: `{end_of_turn}`; user->assistant handoff string: `{handoff}`",
              f"- assistant-slot opening beyond the stored-turn header (assistant_opening_extra): `{render.assistant_opening_extra(tok)}`",
              f"- config: dtype {st['dtype']}, offload {st['offload']} (gpu_gib {st['gpu_gib']}), scoring batch size {st['batch_size']}, "
              f"generation precision {st['gen_precision']}, extra handoff markers {st['handoff_extra']}, "
              f"assistant_think_block {st['assistant_think_block']}, contamination check possible: {fam['contamination_check']}"]

    demo = tok.apply_chat_template(DEMO, tokenize=False)
    lines += ["", f"Special tokens: bos={tok.bos_token!r} eos={tok.eos_token!r} pad={tok.pad_token!r}", "",
              "Instruct template rendering of a user + assistant exchange:", "", "```", demo, "```", "",
              "Instruct chat template:", "", "```jinja", templates[ref_stage] or "", "```", ""]
    assert same_render, f"{name}: post-trained chat templates render differently"
    assert same_merges and only_reserved, f"{name}: tokenizers differ beyond reserved tokens"
    assert verbatim is True, f"{name}: chat template does not place content verbatim"
    return lines


def write_models_md(name, lines, path="docs/models.md"):
    """The default family owns the top of the file; others live in marked sections that are replaced in place."""
    import os
    begin, end = f"<!-- family:{name} -->", f"<!-- /family:{name} -->"
    if name == config.FAMILY_DEFAULT_NAME:
        old = open(path).read() if os.path.exists(path) else ""
        rest = old[old.index("<!-- family:"):] if "<!-- family:" in old else ""
        open(path, "w").write("\n".join(lines) + ("\n" + rest if rest else ""))
        return
    old = open(path).read() if os.path.exists(path) else ""
    block = "\n".join([begin, *lines, end]) + "\n"
    if begin in old:
        old = old[: old.index(begin)] + block + old[old.index(end) + len(end) + 1:]
    else:
        old = old.rstrip("\n") + "\n\n" + block
    open(path, "w").write(old)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=config.FAMILY_DEFAULT_NAME, help="a config.FAMILIES key, or 'all'")
    args = ap.parse_args()
    names = list(config.FAMILIES) if args.family == "all" else [args.family]
    for name in names:
        lines = check_family(name)
        write_models_md(name, lines)
        print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
