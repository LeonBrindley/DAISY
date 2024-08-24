#!/bin/bash

# Define the path to the Python executable and the training script
PYTHON_PATH="/home/omsst2/jupyter-env/bin/python"
SCRIPT_PATH="train_experiment.py"

# Define the list of model names
MODELS=("resnet" "plantnet" "inaturalist" "inceptionv3" "alexnet" "vgg16" "densenet121")

# Loop over each model and run the training script with different configurations
for MODEL in "${MODELS[@]}"; do
    $PYTHON_PATH $SCRIPT_PATH --model_name "$MODEL" --unfreeze_layers_count "1" --use_extra_perceptron "False" --learning_rate "0.001"
    $PYTHON_PATH $SCRIPT_PATH --model_name "$MODEL" --unfreeze_layers_count "2" --use_extra_perceptron "False" --learning_rate "0.001"
    $PYTHON_PATH $SCRIPT_PATH --model_name "$MODEL" --unfreeze_layers_count "1" --use_extra_perceptron "True" --learning_rate "0.001"
    $PYTHON_PATH $SCRIPT_PATH --model_name "$MODEL" --unfreeze_layers_count "2" --use_extra_perceptron "True" --learning_rate "0.001"
done

