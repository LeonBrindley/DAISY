import data
import resnet
import plantnet
import inaturalist
import inceptionv3
import alexnet
import vgg16
import densenet121
import simplecnn
import deepverge
import ensemble
import validate
import serialisation
import argparse
import re

#! GLOBALS !#

epochs = 100
patience = 5

#! END GLOBALS !#

def model_path_to_model_name(model_path):
    model_names = ["resnet", "plantnet", "inaturalist", "inceptionv3", "alexnet", "vgg16", "densenet121", "simplecnn", "deepverge"]

    for model_name in model_names:
        if model_name in model_path:
            return model_name
        
    raise ValueError("wah")
    
model_name_input_shapes = {
    "resnet": (224,224),
    "plantnet": (224,224),
    "inaturalist": (224,224),
    "inceptionv3": (299, 299),
    "alexnet": (224,224),
    "vgg16": (224,224),
    "densenet121": (224,224),
    "simplecnn": (224,224),
    "deepverge": (384,384)
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an ensemble.')

    parser.add_argument('--model_paths', type=str, required=True, 
                        help='Paths to models to use in the ensemble. Comma separated.')

    parser.add_argument('--ensemble_size', type=str, required=True, 
                        help='Size of ensemble, passed as a string (0, 1 or 2).')

    parser.add_argument('--learning_rate', type=str, required=True, 
                        help='Learning rate, passed as a string representation of a float.')
    
    parser.add_argument('--num_models', type=str, required=True, 
                        help='Number of models.')
    
    parser.add_argument('--dropout_p', type=str, default="None", 
                        help='Dropout probability, passed as a string representation of a float. None if no dropout.')
    
    parser.add_argument('--weight_decay', type=str, default="None", 
                        help='Weight decay for L2 regularisation, passed as a string representation of a float.')

    args = parser.parse_args()
    return args

def load_model(model_path):
    use_extra_perceptron = "_pTrue" in model_path
    dp_match = re.search(r'_dp([0-9.e-]+|None)', model_path)
    dropout_p = float(dp_match.group(1)) if dp_match and dp_match.group(1) != 'None' else None

    if "resnet" in model_path:
        base_model = resnet.load(use_extra_perceptron, dropout_p)
    elif "plantnet" in model_path:
        base_model = plantnet.load(use_extra_perceptron, dropout_p)
    elif "inaturalist" in model_path:
        base_model = inaturalist.load(use_extra_perceptron, dropout_p)
    elif "inceptionv3" in model_path:
        base_model = inceptionv3.load(use_extra_perceptron, dropout_p)
    elif "alexnet" in model_path:
        base_model = alexnet.load(use_extra_perceptron, dropout_p)
    elif "vgg16" in model_path:
        base_model = vgg16.load(use_extra_perceptron, dropout_p)
    elif "densenet121" in model_path:
        base_model = densenet121.load(use_extra_perceptron, dropout_p)
    elif "simplecnn" in model_path:
        base_model = simplecnn.load(dropout_p)
    elif "deepverge" in model_path:
        base_model = deepverge.load(dropout_p)
    else:
        raise ValueError("wah")

    model = serialisation.load_only_model(model_path, base_model)
    return model

def main():
    args = parse_arguments()

    # Convert string arguments to appropriate types
    models_paths = args.model_paths.split(',')
    ensemble_size = int(args.ensemble_size)
    learning_rate = float(args.learning_rate)
    num_models = int(args.num_models)
    dropout_p =  None if args.dropout_p == "None" else float(args.dropout_p)
    weight_decay = None if args.weight_decay == "None" else float(args.weight_decay)

    print(f"Model Paths: {models_paths}")
    print(f"Ensemble Size: {ensemble_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Num Models: {num_models}")
    print(f"Dropout Probability: {dropout_p}")
    print(f"Weight Decay: {weight_decay}")
    
    # Load data
    train_loader, val_loader, train_dataset, val_dataset, label_map = data.load_data((224, 224), "~/gp/dataV6_aug.csv")

    models_paths = models_paths[:num_models]
    model_names = [model_path_to_model_name(model_path) for model_path in models_paths]

    print(f"Model names:\n{model_names}")
    print(f"Model paths:\n{models_paths}")

    # Load base models
    models = [load_model(model_path) for model_path in models_paths]
    input_sizes = [model_name_input_shapes[model_name] for model_name in model_names]

    # Train model
    model, train_stats, val_stats = ensemble.train_ensemble(train_loader, val_loader, models, input_sizes, ensemble_size, weight_decay, dropout_p, learning_rate, epochs)

    # Perform validation inference
    val_predictions = validate.validation_inference(model, val_dataset)

    # Save the results
    model_dir = f"/rds/user/omsst2/hpc-work/gp/results_v6_ensemble/ensemble_nm{num_models}_s{ensemble_size}_dp{dropout_p}_wd{weight_decay}_lr{learning_rate}"
    serialisation.save_model(model, train_stats, val_stats, val_predictions, model_dir)

if __name__ == "__main__":
    main()



# Best models
# resnet_uf3_pTrue_lr0.0001
# plantnet_uf3_pFalse_lr0.0001
# vgg16_uf6_pTrue_lr0.0001
# densenet121_uf5_pTrue_lr0.0001
# alexnet_uf5_pTrue_lr0.0001
# inaturalist_uf3_pTrue_lr0.0001