#!/bin/zsh
# contrastive objective: cosine("this sound is a <word> cello") - cosine("this sound is a cello"); cello first, after the named grid
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
while pgrep -f run_named.sh > /dev/null; do sleep 30; done
F='UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated'
for INST in cello brass epiano flute; do
  T2_NAMED=1 T2_OBJ=contrast nice -n 10 .venv/bin/python experiments/text2fx/words/word_grid.py $INST 2>&1 | grep --line-buffered -v "$F" | grep --line-buffered -v "^\s*$"
done
echo "CONTRAST GRID DONE"
