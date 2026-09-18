#!/bin/zsh
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
until grep -q "BALANCE DONE" experiments/text2fx/words/run_balance.log; do sleep 30; done
F='UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated'
echo "=== variant D"
nice -n 10 .venv/bin/python experiments/text2fx/words/balance_test.py D 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"
echo "D DONE"
