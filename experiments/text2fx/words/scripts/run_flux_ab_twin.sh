#!/bin/zsh
# after the repeatability run finishes: the flux A/B on the cell where the hinge actually engaged (e-piano dark, twin route)
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
while pgrep -f "ladder_test.py cello:punchy fx" > /dev/null; do sleep 10; done
T2_AB_CELL=epiano:dark T2_AB_ROUTE=twin nice -n 10 .venv/bin/python experiments/text2fx/words/flux_ab.py
echo "TWIN AB DONE"
