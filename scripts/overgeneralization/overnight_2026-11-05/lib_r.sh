# shared helpers for the spec R chains (source me). PY, repo cwd assumed.
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
# free a model's weights only when disk is tight (less than $2 GB free)
free_if_tight() { local avail; avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -lt "${2:-45}" ] && free_model "$1"; return 0; }
stage_model() { $PY -c "from core import config; config.use_family('$1'); print(config.MODELS['$2'])"; }
# retry a command at most twice more; log failures to $FAILLOG
retry() { local n=0; until "$@"; do n=$((n+1)); echo "FAILED($n): $*" | tee -a "${FAILLOG:-/tmp/claude-0/r_failures.log}"; [ $n -ge 3 ] && return 1; sleep 20; done; }
# commit given paths and push to main; one at a time (flock), retries on a busy remote
push_paths() { local msg="$1"; shift
  ( flock -x 8
    git add "$@" 2>/dev/null; git commit -q -m "$msg

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || { echo "nothing to commit: $msg"; exit 0; }
    for i in 1 2 3; do git fetch -q && git pull -q --rebase --autostash && nice -n 15 git push -q origin HEAD:main && { echo "pushed: $msg"; exit 0; }; sleep 15; done; echo "PUSH FAILED: $msg"; exit 1 ) 8>/tmp/claude-0/push.lock; }
gpu_wait() { while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
  echo "owner=$1 purpose=$2" > /tmp/claude-0/gpu.lock; }
gpu_release() { rm -f /tmp/claude-0/gpu.lock; }
