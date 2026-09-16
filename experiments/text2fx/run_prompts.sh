#!/bin/zsh
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
SRC=experiments/text2fx/guitar_folk.wav
for GS in 0.0 0.05; do
  for P in "this sound is underwater" "this sound is woody" "this sound is church bells ringing" \
           "this sound is an african dance on fire" "this sound is folkish and melodic"; do
    .venv/bin/python experiments/text2fx/grafx_chain.py $GS "$P" $SRC guitar 2>&1 \
      | grep -v "UserWarning\|warnings.warn\|Loading weights\|HF_TOKEN\|unauthenticated" | grep -vE "^\s+seed[0-9]"
  done
done
echo "ALL DONE"
