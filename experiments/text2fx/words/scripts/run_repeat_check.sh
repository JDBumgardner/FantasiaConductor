#!/bin/zsh
# the repeatability proof: the same ladder cell twice on each route; stops.json must be identical
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
for arm in a b; do
  echo "[$arm] cello:punchy fx"; T2_LADDER_OUT=$PWD/experiments/text2fx/words/ladder_det/$arm nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py cello:punchy fx
done
for arm in a b; do
  echo "[$arm] epiano:dark twin"; T2_LADDER_OUT=$PWD/experiments/text2fx/words/ladder_det/$arm nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py epiano:dark twin
done
echo "REPEAT CHECK DONE"
