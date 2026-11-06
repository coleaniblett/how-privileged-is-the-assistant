#!/usr/bin/env bash
# Spec S Stage 0 step 1: fetch third-party sources at pinned commits into $BA_SRC (default /root/big_atlas_src, OUTSIDE the repo).
# assistant-axis and Prompting-with-Social-Roles state no licence and this repo is public: their text must never be committed.
set -e
BA_SRC="${BA_SRC:-/root/big_atlas_src}"; mkdir -p "$BA_SRC"; cd "$BA_SRC"
fetch() { local url="$1" pin="$2" d; d=$(basename "$url"); [ -d "$d" ] || git clone -q "https://github.com/$url" "$d"; git -C "$d" fetch -q origin "$pin" 2>/dev/null || true; git -C "$d" checkout -q "$pin"; echo "$d $(git -C "$d" rev-parse HEAD)"; }
fetch safety-research/assistant-axis a98961956072224eaf244eb289d6c01700b63795
fetch safety-research/persona_vectors b8e0f044fe2410a6fad579f38324f03f13b4e917
fetch Jiaxin-Pei/Prompting-with-Social-Roles 9f0ab13dcea5260554eab72eb124d4c7557881e1
fetch anthropics/evals 84fcc677e52e1902d696c32cd1a6b663e70d3993
