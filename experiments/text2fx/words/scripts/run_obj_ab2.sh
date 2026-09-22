#!/bin/zsh
# the directional objective with two-sided stops (the stop sets the amount), after the first batch finishes
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
while pgrep -f "run_obj_ab.sh" > /dev/null; do sleep 15; done
for job in "epiano:dark twin" "cello:punchy fx" "cello:soft fx" "cello:soft twin"; do
  echo "[direq] $job"; T2_LADDER_OBJ=dir T2_LADDER_EQ=1 T2_LADDER_OUT=$PWD/experiments/text2fx/words/ladder_obj/direq nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py ${=job}
done
echo "OBJ AB2 DONE"
