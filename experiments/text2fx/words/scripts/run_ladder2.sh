#!/bin/zsh
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
F='UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated'
for JOB in "cello:punchy fx" "cello:punchy twin" "cello:soft twin"; do
  nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py ${=JOB} 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"
done
echo "LADDER2 DONE"
