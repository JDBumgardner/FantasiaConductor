#!/bin/zsh
# the named-prompt grid ("this sound is a dark cello"); waits for the plain grid to release the GPU, cello first
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
while pgrep -f run_grid.sh > /dev/null; do sleep 30; done
F='UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated'
for INST in cello brass epiano flute; do
  T2_NAMED=1 nice -n 10 .venv/bin/python experiments/text2fx/words/word_grid.py $INST 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"
done
echo "NAMED GRID DONE"
