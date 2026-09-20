#!/bin/zsh
# second pair of the e-piano dark twin flux A/B (same seeds; MPS nondeterminism gives a second sample of each arm)
cd /Users/jacobbumgardner/ClaudeProjects/FantasiaCondutcor
for arm in nograd fixed; do
  NG=0; [[ $arm == nograd ]] && NG=1
  echo "[$arm rep] start"
  T2_FLUX_NOGRAD=$NG T2_LADDER_OUT=$PWD/experiments/text2fx/words/ladder_flux/${arm}_rep nice -n 10 .venv/bin/python experiments/text2fx/words/ladder_test.py epiano:dark twin
  echo "[$arm rep] done"
done
echo "TWIN AB REP DONE"
