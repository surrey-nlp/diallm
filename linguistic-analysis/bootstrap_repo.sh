#!/usr/bin/env bash
# Move this project to ~/workspace/dialllm-analysis and create a PRIVATE GitHub
# repo under the dipteshkanojia account, then push.
#
# Prereqs: git, and GitHub CLI `gh` authenticated as dipteshkanojia
#   (run `gh auth status`; if needed `gh auth login`).
#
# Run from inside this folder:  bash bootstrap_repo.sh
set -euo pipefail

DEST="$HOME/workspace/dialllm-analysis"
SRC="$(pwd)"

mkdir -p "$HOME/workspace"
if [ "$SRC" != "$DEST" ]; then
  echo "Moving project to $DEST ..."
  mkdir -p "$DEST"
  # copy everything except local data/results and any existing .git
  rsync -a --exclude '.git' --exclude 'data' --exclude 'results' "$SRC"/ "$DEST"/
  cd "$DEST"
else
  cd "$DEST"
fi

if [ ! -d .git ]; then
  git init
  git add .
  git commit -m "DiaLLM linguistic analysis: detectors, measures, bridge, plan"
fi

# Create the private repo and push (requires authenticated gh).
gh repo create dipteshkanojia/dialllm-analysis --private --source=. --remote=origin --push

echo "Done. Private repo: https://github.com/dipteshkanojia/dialllm-analysis"
