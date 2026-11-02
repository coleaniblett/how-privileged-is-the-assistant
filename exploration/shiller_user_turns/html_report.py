"""One self-contained HTML page for reading the open-user-turn continuations of several models side by side.

    python -m exploration.shiller_user_turns.html_report [--models gemma4-31b-it,qwen3-8b] [--out ...]

Reads results/exploration/shiller_user_turns/<model>.jsonl (stdlib only; no GPU) and, if present, the pilot's hand
labels for Qwen3-8B (measures/vestigial_regression/pilot/qwen3-8b_generation_labels.csv). Writes continuations.html next
to the data. The page embeds all samples as JSON; filters, search and the flag chips run in the browser.
"""

import argparse
import csv
import json
import os

OUT = "results/exploration/shiller_user_turns"  # same as generate.OUT, without importing torch

PILOT_LABELS = {"qwen3-8b": "measures/vestigial_regression/pilot/qwen3-8b_generation_labels.csv"}
KEEP = ["model", "condition", "prompt_id", "sample", "prefill", "completion", "user_portion", "n_new_tokens",
        "stopped_on_eos", "ended_turn", "handoff_to_assistant", "new_user_turn", "assistant_register",
        "assistant_register_unmarked", "plain_role_marker", "assistant_thinking", "special_token",
        "repetition_loop", "token_loop"]
FLAGS = ["ended_turn", "handoff_to_assistant", "new_user_turn", "assistant_register", "assistant_thinking",
         "plain_role_marker", "special_token", "repetition_loop", "token_loop"]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open user-turn continuations</title>
<style>
:root{--bg:#fbfaf8;--fg:#1f2328;--muted:#59636e;--card:#ffffff;--line:#d9d6d0;--user:#e8f1e4;--rest:#f1ece4;
--chip:#e7e4ee;--chipfg:#3b2f63;--flag:#f6e2d3;--flagfg:#7a3410;--good:#dcefe0;--goodfg:#1e5a2e;--bad:#f6dcdc;--badfg:#8a1f1f;
--accent:#2b5fa8;--g:#8a4b14;--q:#1d5e63}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16171a;--fg:#e6e6e6;--muted:#9aa3ad;--card:#1e2024;
--line:#383b41;--user:#22331f;--rest:#2b2620;--chip:#2e2a40;--chipfg:#cbc2f0;--flag:#3d2a1c;--flagfg:#f0b88e;--good:#1e3324;
--goodfg:#9fd8ac;--bad:#3b2020;--badfg:#f0a8a8;--accent:#79a8e8;--g:#e0a064;--q:#6fc4ca}}
:root[data-theme=dark]{--bg:#16171a;--fg:#e6e6e6;--muted:#9aa3ad;--card:#1e2024;--line:#383b41;--user:#22331f;--rest:#2b2620;
--chip:#2e2a40;--chipfg:#cbc2f0;--flag:#3d2a1c;--flagfg:#f0b88e;--good:#1e3324;--goodfg:#9fd8ac;--bad:#3b2020;--badfg:#f0a8a8;
--accent:#79a8e8;--g:#e0a064;--q:#6fc4ca}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px;border-bottom:1px solid var(--line)}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin:0 0 12px;max-width:80ch}
.controls{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;
flex-wrap:wrap;gap:10px 18px;align-items:center}
.controls label{font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center}
select,input[type=search]{font:inherit;font-size:13px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
input[type=search]{min-width:200px}
.flags{display:flex;flex-wrap:wrap;gap:4px 10px}
.flags label{color:var(--fg)}
main{padding:16px;max-width:1500px;margin:0 auto}
table.rates{border-collapse:collapse;font-size:13px;margin:8px 0 0}
table.rates th,table.rates td{border:1px solid var(--line);padding:3px 8px;text-align:right}
table.rates th:first-child,table.rates td:first-child{text-align:left}
.prompt{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:0 0 18px;overflow:hidden}
.prompt>.head{padding:10px 14px;border-bottom:1px solid var(--line)}
.pid{font-size:12px;color:var(--muted)}
.opener{margin:4px 0 0;white-space:pre-wrap;font-weight:600}
.opener.empty{font-weight:400;color:var(--muted);font-style:italic}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.col{padding:10px 14px;border-right:1px solid var(--line);min-width:0}
.col:last-child{border-right:0}
.col h3{margin:0 0 8px;font-size:13px;letter-spacing:.02em}
.col h3.gemma4-31b-it{color:var(--g)}.col h3.qwen3-8b{color:var(--q)}
.sample{border:1px solid var(--line);border-radius:8px;margin:0 0 10px;overflow:hidden}
.sample .meta{display:flex;flex-wrap:wrap;gap:4px;align-items:center;padding:5px 8px;border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}
.txt{margin:0;padding:8px 10px;white-space:pre-wrap;word-break:break-word;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:15em;overflow:auto}
.txt.expanded{max-height:none}
.u{background:var(--user)}.r{background:var(--rest)}
.pre{opacity:.75}
.chip{display:inline-block;background:var(--chip);color:var(--chipfg);border-radius:4px;padding:0 4px;margin:0 1px;font-size:11px}
.fchip{background:var(--flag);color:var(--flagfg);border-radius:10px;padding:0 7px;font-size:11px}
.lab-good{background:var(--good);color:var(--goodfg);border-radius:10px;padding:0 7px;font-size:11px}
.lab-bad{background:var(--bad);color:var(--badfg);border-radius:10px;padding:0 7px;font-size:11px}
button.more{border:0;background:none;color:var(--accent);cursor:pointer;font-size:12px;padding:0 4px;margin-left:auto}
.legend{font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap;margin-top:8px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px;border:1px solid var(--line)}
.none{color:var(--muted);padding:30px;text-align:center}
@media (max-width:600px){.col{border-right:0;border-bottom:1px solid var(--line)}}
</style>
</head>
<body>
<header>
<h1>Open user-turn continuations</h1>
<p class="sub">Each prompt is the opener of an unterminated user turn; the model continues it. Green is the <b>user portion</b>
(text before the first end-of-turn, turn-open or stop token); brown is whatever follows it. Purple chips are special tokens
kept from the raw completion. Orange chips are the exploration's heuristic flags, not ground truth.</p>
<div class="legend"><span><span class="sw" style="background:var(--user)"></span>user portion</span>
<span><span class="sw" style="background:var(--rest)"></span>after the user portion</span>
<span><span class="chip">&lt;|channel&gt;</span> special token</span>
<span><span class="fchip">flag</span> heuristic flag</span>
<span><span class="lab-good">user voice</span> <span class="lab-bad">not user voice</span> hand label (Qwen3-8B, 20 samples)</span></div>
<div id="rates"></div>
</header>
<div class="controls">
<label>Condition <select id="cond"></select></label>
<label>Prompt <select id="prompt"></select></label>
<label>Search <input type="search" id="q" placeholder="text in opener or completion"></label>
<label><input type="checkbox" id="onlyflagged"> only samples with a selected flag</label>
<label><input type="checkbox" id="expand"> expand all</label>
<label>Theme <select id="theme"><option value="">auto</option><option value="light">light</option><option value="dark">dark</option></select></label>
<div class="flags" id="flags"></div>
</div>
<main id="out"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const ROWS = D.rows, FLAGS = D.flags, MODELS = D.models, LABELS = D.labels;
const $ = id => document.getElementById(id);
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const render = s => esc(s).replace(/&lt;(\|[^&\s]+|[^&\s|]+\||\/?think|bos|eos|pad)&gt;/g, m => '<span class="chip">'+m+'</span>');
const conds = [...new Set(ROWS.map(r=>r.condition))];
$('cond').innerHTML = conds.map(c=>`<option>${c}</option>`).join('');
$('cond').value = conds.includes('ordinary') ? 'ordinary' : conds[0];
$('flags').innerHTML = FLAGS.map(f=>`<label><input type="checkbox" class="fl" value="${f}">${f}</label>`).join('');
function fillPrompts(){
  const ids = [...new Set(ROWS.filter(r=>r.condition===$('cond').value).map(r=>r.prompt_id))];
  $('prompt').innerHTML = '<option value="">all prompts</option>' + ids.map(i=>`<option>${i}</option>`).join('');
}
function rates(){
  const rows = ['<tr><th>model / condition</th><th>n</th>'+FLAGS.map(f=>`<th>${f}</th>`).join('')+'</tr>'];
  for (const m of MODELS) for (const c of conds){
    const rs = ROWS.filter(r=>r.model===m && r.condition===c); if(!rs.length) continue;
    rows.push(`<tr><td>${m} / ${c}</td><td>${rs.length}</td>`+FLAGS.map(f=>`<td>${Math.round(100*rs.filter(r=>r[f]).length/rs.length)}%</td>`).join('')+'</tr>');
  }
  $('rates').innerHTML = '<table class="rates">'+rows.join('')+'</table>';
}
function sample(r){
  const up = r.user_portion || '';
  const hasSplit = r.completion.startsWith(up);
  const head = hasSplit ? up : r.completion, rest = hasSplit ? r.completion.slice(up.length) : '';
  const on = FLAGS.filter(f=>r[f]).map(f=>`<span class="fchip">${f}</span>`).join('');
  const lab = LABELS[[r.model,r.prompt_id,r.sample].join('|')];
  const labHtml = lab ? `<span class="${lab==='user voice'?'lab-good':'lab-bad'}">${lab}</span>` : '';
  return `<div class="sample"><div class="meta"><b>#${r.sample}</b><span>${r.n_new_tokens} tok${r.stopped_on_eos?', ended at eos':''}</span>${labHtml}${on}<button class="more">expand</button></div>`
    + `<pre class="txt ${$('expand').checked?'expanded':''}"><span class="pre">${esc(r.prefill)}</span><span class="u">${render(head)}</span>${rest?`<span class="r">${render(rest)}</span>`:''}</pre></div>`;
}
function draw(){
  const cond = $('cond').value, pid = $('prompt').value, q = $('q').value.toLowerCase();
  const sel = [...document.querySelectorAll('.fl:checked')].map(e=>e.value), onlyFlagged = $('onlyflagged').checked;
  let rs = ROWS.filter(r=>r.condition===cond && (!pid || r.prompt_id===pid));
  if (q) rs = rs.filter(r=>(r.prefill+' '+r.completion).toLowerCase().includes(q));
  if (onlyFlagged && sel.length) rs = rs.filter(r=>sel.some(f=>r[f]));
  const byPrompt = new Map();
  for (const r of rs){ if(!byPrompt.has(r.prompt_id)) byPrompt.set(r.prompt_id, []); byPrompt.get(r.prompt_id).push(r); }
  if (!byPrompt.size){ $('out').innerHTML = '<div class="none">No samples match.</div>'; return; }
  $('out').innerHTML = [...byPrompt].map(([id, list])=>{
    const pre = list[0].prefill;
    const cols = MODELS.map(m=>{
      const ms = list.filter(r=>r.model===m).sort((a,b)=>a.sample-b.sample);
      return `<div class="col"><h3 class="${m}">${m} (${ms.length} shown)</h3>${ms.map(sample).join('')||'<div class="none">none</div>'}</div>`;
    }).join('');
    return `<section class="prompt"><div class="head"><div class="pid">${cond} / ${id}</div><div class="opener ${pre?'':'empty'}">${pre?esc(pre):'(empty: the user header and nothing else)'}</div></div><div class="cols">${cols}</div></section>`;
  }).join('');
}
$('out').addEventListener('click', e=>{ if(e.target.classList.contains('more')){ const t=e.target.closest('.sample').querySelector('.txt'); t.classList.toggle('expanded'); e.target.textContent=t.classList.contains('expanded')?'collapse':'expand'; }});
$('cond').onchange = ()=>{ fillPrompts(); draw(); };
for (const id of ['prompt','q','onlyflagged','expand']) $(id).addEventListener(id==='q'?'input':'change', draw);
$('flags').addEventListener('change', draw);
$('theme').onchange = e=>{ if(e.target.value) document.documentElement.dataset.theme=e.target.value; else delete document.documentElement.dataset.theme; };
rates(); fillPrompts(); draw();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma4-31b-it,qwen3-8b")
    ap.add_argument("--dir", default=OUT)
    ap.add_argument("--out")
    args = ap.parse_args()
    models = args.models.split(",")
    rows, labels = [], {}
    for m in models:
        with open(os.path.join(args.dir, f"{m}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({k: r.get(k) for k in KEEP})
        lp = PILOT_LABELS.get(m)
        if lp and os.path.exists(lp):
            with open(lp, newline="", encoding="utf-8") as f:
                for lab in csv.DictReader(f):
                    labels[f"{m}|{lab['prompt_id']}|{lab['sample']}"] = lab["label"]
    rows.sort(key=lambda r: (r["condition"], r["prompt_id"], r["model"], r["sample"]))
    data = json.dumps(dict(rows=rows, flags=FLAGS, models=models, labels=labels), ensure_ascii=False)
    data = data.replace("</", "<\\/")
    out = args.out or os.path.join(args.dir, "continuations.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", data))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB, {len(rows)} samples, {len(labels)} labels)")


if __name__ == "__main__":
    main()
