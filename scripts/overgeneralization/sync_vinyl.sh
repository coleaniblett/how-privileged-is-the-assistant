#!/usr/bin/env bash
# copy the current code and small data files to vinyl's worktree (no deletes; never touches /root/g4nla)
cd /root/hpa-main-wt
rsync -a -e ssh extras/ arena9-vinyl:/root/hpa-atlas2/extras/ --exclude='__pycache__'
rsync -a -e ssh scripts/overgeneralization/ arena9-vinyl:/root/hpa-atlas2/scripts/overgeneralization/
rsync -a -e ssh data/ arena9-vinyl:/root/hpa-atlas2/data/ --exclude='*.parquet'
rsync -a -e ssh measures/overgeneralization/*.py arena9-vinyl:/root/hpa-atlas2/measures/overgeneralization/
