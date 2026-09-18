#!/bin/zsh
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
F='UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated'
for V in E F; do echo "=== variant $V"; nice -n 10 .venv/bin/python experiments/text2fx/words/balance_test.py $V 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"; done
echo "EF DONE"
