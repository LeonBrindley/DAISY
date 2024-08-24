#!/bin/bash

# Define the path to the Python executable and the training script
PYTHON_PATH="/home/omsst2/jupyter-env/bin/python"
SCRIPT_PATH="train_experiment.py"

# Define the list of model names

$PYTHON_PATH $SCRIPT_PATH --model_name "simplecnn" --learning_rate "0.0001" --use_dropout="False" --weight_decay="None"
$PYTHON_PATH $SCRIPT_PATH --model_name "simplecnn" --learning_rate "0.0001" --use_dropout="True" --weight_decay="None"
$PYTHON_PATH $SCRIPT_PATH --model_name "simplecnn" --learning_rate "0.0001" --use_dropout="False" --weight_decay="1e-5"
$PYTHON_PATH $SCRIPT_PATH --model_name "deepverge" --learning_rate "0.0001"

