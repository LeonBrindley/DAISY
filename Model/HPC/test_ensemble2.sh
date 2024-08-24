#!/bin/bash

# Define the path to the Python executable and the training script
PYTHON_PATH="/home/omsst2/jupyter-env/bin/python"
SCRIPT_PATH="ensemble_experiment.py"

# Define the list of model names

$PYTHON_PATH $SCRIPT_PATH --model_paths "/rds/user/omsst2/hpc-work/gp/results_v6_grid/inaturalist_uf5_pFalse_dp0.1_wd1e-05_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/densenet121_uf7_pTrue_dp0.3_wd1e-05_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/inceptionv3_uf10_pTrue_dp0.3_wd0.01_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/alexnet_uf7_pTrue_dpNone_wd0.0015_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/vgg16_uf6_pTrue_dpNone_wd0.01_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/deepverge_uf0_pFalse_dp0.5_wdNone_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/simplecnn_uf0_pFalse_dp0.1_wd0.01_lr0.0001" --ensemble_size "0" --num_models "7" --learning_rate "0.001"

