#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate coastal-monitoring
git pull
./update_change_detection.py
jupyter nbconvert --to notebook --execute --inplace apply_tidal_correction.ipynb calc_linear_trends.ipynb 
git commit -am 'auto update' --author="coastal-monitor <ubuntu@monitoring>"
git push
conda deactivate
