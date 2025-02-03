#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate coastal-monitoring
git pull
./download_images.py > .update_downloads.log
./process_change_detection.py > .update_processing.log
jupyter nbconvert --to notebook --execute --inplace apply_tidal_correction.ipynb calc_linear_trends.ipynb 
git commit -am 'auto update' --author="coastal-monitor <ubuntu@monitoring>"
git push
conda deactivate
