#!/bin/zsh
# Main fine-tune + 3 comparisons, then test-set evaluation. Resumable: rerun to continue.
cd ~/Documents/PlanetFinders
for mode in pretrained random frozen none; do
  run=runs/ft_$mode
  if [ -f $run/DONE ]; then echo "$run done, skipping"; continue; fi
  python3 -u finetune.py --run $run --encoder $mode > $run.log 2>&1 && touch $run/DONE || { echo "$run FAILED"; exit 1; }
  echo "$run finished: $(grep '^done' $run.log)"
done
python3 -u evaluate.py runs/ft_pretrained runs/ft_random runs/ft_frozen runs/ft_none > runs/ft_eval.log 2>&1 && echo "evaluation written"
