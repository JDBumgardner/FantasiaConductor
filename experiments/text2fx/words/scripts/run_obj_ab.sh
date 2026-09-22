#!/bin/zsh
# directional objective vs cosine, same cells, deterministic pipeline
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
for job in "epiano:dark twin" "cello:punchy fx" "cello:soft fx" "cello:soft twin"; do
  echo "[dir] $job"; T2_LADDER_OBJ=dir T2_LADDER_OUT=$PWD/experiments/text2fx/words/ladder_obj/dir nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py ${=job}
done
for job in "cello:soft fx" "cello:soft twin"; do
  echo "[cos] $job"; T2_LADDER_OBJ=cos T2_LADDER_OUT=$PWD/experiments/text2fx/words/ladder_obj/cos nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py ${=job}
done
echo "OBJ AB DONE"
