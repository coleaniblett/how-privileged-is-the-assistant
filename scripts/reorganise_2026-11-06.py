"""Re-apply the 2026-11-06 reorganisation (docs/moved_paths_2026-11-06.md) after merging work done on the old layout.

    python scripts/reorganise_2026-11-06.py            # dry run: what would move and which files would be rewritten
    python scripts/reorganise_2026-11-06.py --apply    # git mv the files, rewrite references; then review `git diff` and commit

Idempotent. Git carries modified files across the moves when merging, but files *added* on the old layout land at old paths (for
example a new results/overnight_2026-11-05/<task>/ or a new scripts/exploratory/ launcher). This moves every tracked file still at an
old path to its new one (git mv), places new launchers by name or by the module they run, rewrites old paths in code, launchers, tests
and docs, and lists what it cannot decide: unknown top-level results directories, launchers it could not place, and code that builds
old-style paths (config.RESULTS_ROOT + family, measure_dir) which needs a hand edit. Raw records (csv, jsonl, json, logs) are not
rewritten. Untracked files on other checkouts: scripts/migrate_untracked_2026-11-06.sh.
"""
import os
import re
import subprocess
import sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
OG = "results/overgeneralization"
EARLY, CARRY, PRE = f"{OG}/early_probes", f"{OG}/persona_carryover", "results/precapstone"
FAMS = ["olmo3-7b", "qwen3-8b", "gemma4-31b", "llama3.1-8b", "apertus-8b"]
PILOTS = ["activation_patching", "activation_patching_j4", "belief_induction", "bias_leak", "dolci_exposure", "dominance_matrix",
          "icl_era", "implant_spread", "mechanism_checks", "orthogonal_steering", "overnight_2026-11-05", "persona_map",
          "persona_suppression", "refusal_direction", "role_says", "second_judge", "skill_roleplay", "style_fall", "training_style",
          "training_style_l4", "trait_atlas", "trait_atlas_breadth", "trait_atlas_incharacter", "trait_atlas_more_base",
          "trait_atlas_strict", "trait_atlas_unmixed", "user_style", "working_model_L", "new_families_j2.md"]
CAPSTONE_OG = {"user_turn_2026-11-06": "results/capstone/overgeneralization/user_turn_2026-11-06"}  # cross-family user-slot summaries
PRE_EXACT = ["scores", "generations", "selection_check", "results.md", "summary.csv"]
PRE_PREFIX = ["role_swap", "coin_toss", "position", "nll_"]
PROBES = ("ai_personas_axis", "assistant_axis", "assistant_steering", "axis_", "carryover_", "decision_point", "dominance_pilot",
          "false_fact", "formality_", "persona_", "plain_axis", "role_", "stakeholder_swap")
DOCS = {"docs/findings_whitebox.md": f"{EARLY}/findings_whitebox.md", "docs/whitebox_arm_proposal.md": f"{EARLY}/whitebox_arm_proposal.md",
        "docs/overnight_report.html": f"{EARLY}/overnight_report.html",
        "docs/overgeneralization_results.html": f"{EARLY}/overgeneralization_results.html",
        "docs/spec_R_overnight.md": f"{OG}/overnight_2026-11-05/spec_R_overnight.md"}
KNOWN_TOP = {"README.md", "capstone", "capstone_think_off", "exploration", "logit_conversations", "overgeneralization", "precapstone"}
LAUNCHERS = "scripts/overgeneralization"
# files that name old paths on purpose; in docs/decisions.md only the reorganisation entry itself is left as written
EXEMPT = {"docs/moved_paths_2026-11-06.md", "scripts/migrate_untracked_2026-11-06.sh", "scripts/reorganise_2026-11-06.py", "attic/README.md"}
OWN_ENTRY = "## 2026-11-06 (UTC) - Repository reorganised by measure and pilot"
# a new launcher goes to the folder of the first module it runs (spec R/T chains by name: run_r8_push.sh, step_t1_vinyl.sh, ...)
MODULE_GROUP = {
    "persona_carryover": "persona_carryover", "user_traits": "user_slot", "exchange_reply_in_user": "user_slot", "judge": "user_slot",
    "user_style": "user_slot", "user_mirror": "user_slot", "user_minimal_pairs": "user_slot", "orthogonal_steering": "orthogonal_steering",
    "leakage_matrix": "dominance_matrix", "persona_map": "persona_map", "implant_spread": "implant_spread", "icl_era": "icl_era",
    "persona_suppression": "persona_suppression", "style_fall_checks": "style_fall", "mechanism_checks": "mechanism_checks",
    "activation_patching": "activation_patching", "activation_patching_j4": "activation_patching", "refusal_direction": "refusal_direction",
    "working_model_l": "working_model_L", "bias_leak": "bias_leak", "training_style": "bias_leak", "skill_roleplay": "skill_roleplay",
    "belief_induction": "belief_induction", "role_says": "role_says", "role_says_breadth": "role_says", "second_judge": "trait_atlas",
    "dolci_exposure": "trait_atlas"}
MODULE_GROUP.update({m: "trait_atlas" for m in ("trait_atlas", "trait_atlas_incharacter", "trait_atlas_more_base", "trait_atlas_strict",
                                               "trait_atlas_breadth", "trait_atlas_unmixed")})
MODULE_GROUP.update({m: "early_probes" for m in (
    "ai_personas_axis", "assistant_axis", "assistant_steering", "axis_gens", "axis_refusal_cos", "axis_transfer", "carryover_balanced",
    "carryover_steer_report", "decision_point", "dominance_pilot", "false_fact", "formality_axis", "introspection_macar",
    "introspection_personas", "persona_axis", "persona_drift", "persona_leakage", "persona_override", "persona_refusal", "plain_axis",
    "role_gradient", "role_register", "stakeholder_swap")})


def new_path(f):
    """The reorganised path of a tracked file, or None if it is not at an old path."""
    p = f.split("/")
    if f in DOCS:
        return DOCS[f]
    if p[0] == "scripts" and len(p) == 3 and p[1] == "exploratory":
        return launcher_path(f)
    if p[0] != "results" or len(p) < 2:
        return None
    top = p[1]
    if top in PRE_EXACT or any(top.startswith(x) for x in PRE_PREFIX):
        return f"{PRE}/{'/'.join(p[1:])}"
    if top in FAMS and len(p) > 2 and p[2] == "overgeneralization":
        return f"{CARRY}/{top}/{'/'.join(p[3:])}"
    if top == "olmo3-7b" and len(p) > 2 and p[2].startswith(PROBES):
        return f"{EARLY}/{'/'.join(p[1:])}"
    if top in ("gemma-4-e4b", "other") or (top in ("llama3.1-8b", "qwen3-8b") and len(p) > 2 and p[2].startswith("introspection_personas")):
        return f"{EARLY}/{'/'.join(p[1:])}"
    if top in PILOTS:
        return f"{OG}/{'/'.join(p[1:])}"
    if top in CAPSTONE_OG:
        return f"{CAPSTONE_OG[top]}/{'/'.join(p[2:])}".rstrip("/")
    return None


UNPLACED = []


def launcher_path(f):
    name = os.path.basename(f)
    if re.match(r"(run|step)_[rt]\d", name) or name in ("lib_r.sh", "queue_driver.sh", "make_morning.py"):
        return f"{LAUNCHERS}/overnight_2026-11-05/{name}"
    try:
        src = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
    except OSError:
        src = ""
    for m in re.findall(r"-m\s+(?:extras|measures\.\w+)\.(\w+)", src):
        if m in MODULE_GROUP:
            return f"{LAUNCHERS}/{MODULE_GROUP[m]}/{name}"
    UNPLACED.append(f)
    return f"{LAUNCHERS}/unsorted/{name}"


def rules():
    """Ordered (pattern, replacement) pairs; each only prefixes complete leading path components, so applying them twice changes nothing."""
    R = [(re.compile(r"(?<![\w])results/(" + re.escape(x) + r")(?![\w-])"), PRE + r"/\1") for x in PRE_EXACT]
    R += [(re.compile(r"(?<![\w])results/(" + re.escape(x) + r")"), PRE + r"/\1") for x in PRE_PREFIX]
    for f in FAMS + ["<family>", "{fam}", "{family}", "$fam", "${fam}"]:
        R.append((re.compile(r"(?<![\w])results/" + re.escape(f) + r"/overgeneralization"), (CARRY + "/" + f).replace("\\", "\\\\")))
    R.append((re.compile(r"(?<![\w])results/olmo3-7b/(?=" + "|".join(PROBES) + ")"), EARLY + "/olmo3-7b/"))
    for m in ("gemma-4-e4b", "other"):
        R.append((re.compile(r"(?<![\w])results/" + re.escape(m) + r"(?=[/`'\" )\]:,.]|$)", re.M), EARLY + "/" + m))
    for m in ("llama3.1-8b", "qwen3-8b"):
        R.append((re.compile(r"(?<![\w])results/" + re.escape(m) + r"/introspection_personas"), f"{EARLY}/{m}/introspection_personas"))
    for x in sorted(PILOTS, key=len, reverse=True):
        R.append((re.compile(r"(?<![\w])results/" + re.escape(x)), OG + "/" + x))
    for x, n in CAPSTONE_OG.items():
        R.append((re.compile(r"(?<![\w])results/" + re.escape(x)), n))
    for o, n in DOCS.items():
        R.append((re.compile(r"(?<![\w/.-])" + re.escape(o) + r"(?![\w.-])"), n))
    return R


def text_files(files):
    for f in files:
        if f in EXEMPT:
            continue
        if f.startswith(("data/", "attic/")) and f not in ("data/README.md", "attic/README.md"):
            continue
        if f.startswith("results/") and not f.endswith((".md", ".html", ".sh", ".py")):
            continue
        if f.endswith((".py", ".sh", ".md", ".html", ".toml", ".txt")) or os.path.basename(f) == ".gitignore":
            yield f


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT, check=True).stdout


def main():
    apply = "--apply" in sys.argv
    files = [f for f in git("ls-files").split("\n") if f]
    moves = [(f, n) for f in files if (n := new_path(f))]
    for f, n in moves:
        print(f"{'mv' if apply else 'would mv'} {f} -> {n}")
        if apply:
            os.makedirs(os.path.join(ROOT, os.path.dirname(n)), exist_ok=True)
            git("mv", f, n)
            if n.startswith(LAUNCHERS + "/") and n.count("/") == 3 and n.endswith(".sh"):
                p = os.path.join(ROOT, n)
                s = open(p).read()
                open(p, "w").write(s.replace('cd "$(dirname "$0")/../.."\n', 'cd "$(dirname "$0")/../../.."\n'))
    files = [f for f in git("ls-files").split("\n") if f] if apply else [dict(moves).get(f, f) for f in files]
    R = rules()
    launcher_names = {os.path.basename(o): n for o, n in moves if o.startswith("scripts/exploratory/")}
    rewritten = []
    for f in text_files(files):
        p = os.path.join(ROOT, f if apply else next((o for o, n in moves if n == f), f))
        if not os.path.isfile(p):
            continue
        s = open(p, encoding="utf-8", errors="surrogateescape").read()
        keep = ""
        if f == "docs/decisions.md" and OWN_ENTRY in s:  # rewrite around the reorganisation entry, never inside it
            a = s.index(OWN_ENTRY)
            b = s.find("\n## ", a + 1)
            b = len(s) if b < 0 else b
            head, keep, tail = s[:a], s[a:b], s[b:]
        else:
            head, tail = s, ""
        parts = []
        for t in (head, tail):
            for pat, r in R:
                t = pat.sub(r, t)
            for name, n in launcher_names.items():
                t = re.sub(r"(?<![\w/])scripts/exploratory/" + re.escape(name) + r"(?![\w.])", n, t)
            t = re.sub(r"(?<![\w/])scripts/exploratory(?=/?[\s`'\")\]]|$)", LAUNCHERS, t, flags=re.M)
            parts.append(t)
        t = parts[0] + keep + parts[1]
        if t != s:
            rewritten.append(f)
            if apply:
                open(p, "w", encoding="utf-8", errors="surrogateescape").write(t)
    print(f"{'rewrote' if apply else 'would rewrite'} references in {len(rewritten)} files" + (": " + ", ".join(rewritten[:30]) if rewritten else ""))
    tops = sorted({f.split("/")[1] for f in files if f.startswith("results/") and f.count("/") >= 1} - KNOWN_TOP)
    if tops:
        print("NOT IN THE LAYOUT (place them and add a rule here): " + ", ".join(f"results/{t}" for t in tops))
    if UNPLACED:
        print("launchers put in scripts/overgeneralization/unsorted/ (move them to their pilot): " + ", ".join(UNPLACED))
    built = []
    for f in files:
        if f.startswith(("extras/", "measures/overgeneralization/")) and f.endswith(".py"):
            p = os.path.join(ROOT, f)
            if os.path.isfile(p) and re.search(r"os\.path\.join\(config\.RESULTS_ROOT, config\.FAMILY|config\.measure_dir\((?!\"extras\")", open(p).read()):
                built.append(f)
    for f in files:  # paths joined from components, e.g. os.path.join(root, "results", "generations") or ROOT / "results" / "x"
        if f.endswith(".py") and f.startswith(("core/", "measures/", "extras/", "tests/")) and f not in built:
            p = os.path.join(ROOT, f)
            src = open(p, encoding="utf-8", errors="replace").read() if os.path.isfile(p) else ""
            parts = re.findall(r"""["']results["']\s*[,/]\s*["']([\w.-]+)["'](?:\s*[,/]\s*["']([\w.-]+)["'])?""", src)
            if any(new_path(f"results/{a}/{b or 'x'}/x") for a, b in parts):
                built.append(f)
    if built:
        print("code that builds old-style result paths (hand edit: see docs/decisions.md 2026-11-06): " + ", ".join(built))


if __name__ == "__main__":
    main()
