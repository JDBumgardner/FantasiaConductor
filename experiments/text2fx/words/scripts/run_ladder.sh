#!/bin/zsh
# one process per cell and route: the MPS driver's untracked pool grows across long runs in one process
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
F='UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated'
for CELL in cello:dark cello:punchy cello:soft epiano:dark; do for R in fx twin; do
  nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py $CELL $R 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"
done; done
echo "LADDER ALL DONE"
