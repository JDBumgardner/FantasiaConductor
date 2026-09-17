#!/bin/zsh
# one process per instrument so the recovery step's leftovers never share a process with the word runs
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
F='UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated'
for INST in cello brass epiano flute; do
  [ -f experiments/text2fx/words/${INST}_patch.json ] || nice -n 10 .venv/bin/python experiments/text2fx/words/word_grid.py recover $INST 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"
  nice -n 10 .venv/bin/python experiments/text2fx/words/word_grid.py $INST 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"
done
echo "GRID DONE"
