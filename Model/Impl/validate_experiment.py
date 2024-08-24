import data
import plantnet
import resnet
import inaturalist
import alexnet
import vgg16
import densenet121
import inceptionv3
import simplecnn
import deepverge
import validate
import serialisation
import argparse

#! GLOBALS !#

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

model_finetune_layers = {
    "resnet": ["fc", "layer4", "layer3", "layer2", "layer1"],
    "plantnet": ["fc", "layer4", "layer3", "layer2", "layer1"],
    "inaturalist": ["fc", "layer4", "layer3", "layer2", "layer1"],
    "inceptionv3": ["inception.fc", "inception.Mixed_7c", "inception.Mixed_7b", "inception.Mixed_7a", "inception.AuxLogits", "inception.Mixed_6e", "inception.Mixed_6d", "inception.Mixed_6c", "inception.Mixed_6b", "inception.Mixed_6a", "inception.Mixed_5d", "inception.Mixed_5c", "inception.Mixed_5b"],
    "alexnet": ["classifier.6", "classifier.4", "classifier.1", "features.10", "features.8", "features.6", "features.3", "features.0"],
    "vgg16": ["classifier.6", "classifier.3", "classifier.0", "features.28", "features.26", "features.24", "features.21"],
    "densenet121": ["classifier", "features.norm5", "features.denseblock4", "features.transition3", "features.denseblock3", "features.transition2", "features.denseblock2", "features.transition1", "features.denseblock1"]
}

# additional fine tune:
# resnet 4 - 5
# plantnet 4 - 5
# inaturalist 4 - 5
# inceptionv3 8 - 13
# alexnet: 6 - 8
# vgg16:
# densenet121: 6 - 9


# Best models in /rds/user/omsst2/hpc-work/gp/results/
# /rds/user/omsst2/hpc-work/gp/results/resnet_uf3_pTrue_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/plantnet_uf3_pFalse_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/vgg16_uf6_pTrue_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/densenet121_uf5_pTrue_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/alexnet_uf5_pTrue_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/inaturalist_uf3_pTrue_lr0.0001
# inceptionv3_uf7_pFalse_lr0.001


# New best models:
# /rds/user/omsst2/hpc-work/gp/results/plantnet_uf3_pFalse_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/inaturalist_uf5_pFalse_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/densenet121_uf7_pTrue_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/resnet_uf3_pTrue_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/vgg16_uf6_pTrue_lr0.0001
# /rds/user/omsst2/hpc-work/gp/results/alexnet_uf7_pTrue_lr0.0001
# inceptionv3_uf10_pTrue_lr0.0001

model_finetune_layers_counts = {
    "resnet": 5, # Best: up to 3
    "plantnet": 5, # Best: up to 3
    "inaturalist": 5, # Best: up to 5
    "inceptionv3": 13, # Best: up to 10
    "alexnet": 8, # Best: up to 7
    "vgg16": 7, # Best: up to 6
    "densenet121": 9 # Best: up to 7
}

epochs = 30
patience = 5

#! END GLOBALS !#

def parse_arguments():
    parser = argparse.ArgumentParser(description='Validate model.')

    parser.add_argument('--model_name', type=str, required=True, 
                        help='Name of the model to be used or saved.')
    parser.add_argument('--unfreeze_layers_count', type=str, default="0", 
                        help='Number of layers to unfreeze, passed as a string (int).')
    parser.add_argument('--use_extra_perceptron', type=str, default="False", 
                        help='Whether to use an extra perceptron, passed as a string ("True" or "False").')
    parser.add_argument('--learning_rate', type=str, required=True, 
                        help='Learning rate, passed as a string representation of a float.')
    parser.add_argument('--use_dropout', type=str, default="False", 
                        help='Use dropout, passed as a string representation of a boolean.')
    parser.add_argument('--weight_decay', type=str, default="None", 
                        help='Weight decay for L2 regularisation, passed as a string representation of a float.')

    args = parser.parse_args()
    return args


def load(model_name, use_extra_perceptron, use_dropout):

    if model_name == "resnet":
        return resnet.load(use_extra_perceptron)

    elif model_name == "plantnet":
        return plantnet.load(use_extra_perceptron)

    elif model_name == "inaturalist":
        return inaturalist.load(use_extra_perceptron)

    elif model_name == "inceptionv3":
        return inceptionv3.load(use_extra_perceptron)

    elif model_name == "alexnet":
        return alexnet.load(use_extra_perceptron)

    elif model_name == "vgg16":
        return vgg16.load(use_extra_perceptron)

    elif model_name == "densenet121":
        return densenet121.load(use_extra_perceptron)
    
    elif model_name == "simplecnn":
        return simplecnn.load(use_dropout)

    elif model_name == "deepverge":
        return deepverge.load()

    else:
        raise ValueError("wah")

def calc_model_path(base_dir, model_name, learning_rate, unfreeze_layers_count, use_extra_perceptron, use_dropout, weight_decay):
    model_dir = f"{base_dir}/{model_name}"
    if model_name not in ["simplecnn", "deepverge"]:
        model_dir += f"_uf{unfreeze_layers_count}_p{use_extra_perceptron}"
    elif model_name == "simplecnn":
        model_dir += f"_dr{use_dropout}_wd{weight_decay}"
    model_dir += f"_lr{learning_rate}"
    return model_dir


def main():
    args = parse_arguments()

    # Convert string arguments to appropriate types
    unfreeze_layers_count = int(args.unfreeze_layers_count)  # Convert string representation of list to actual list
    use_extra_perceptron = args.use_extra_perceptron.lower() == 'true'  # Convert string to boolean
    learning_rate = float(args.learning_rate)  # Convert string to float
    model_name = args.model_name  # String remains a string
    use_dropout = args.use_dropout.lower() == 'true'
    weight_decay = None if args.weight_decay == "None" else float(args.weight_decay)

    if model_name not in model_name_input_shapes:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {list(model_name_input_shapes.keys())}")

    unfreeze_layers = []
    for i in range(unfreeze_layers_count):
        unfreeze_layers.append(model_finetune_layers[model_name][i])

    print(f"Model Name: {model_name}")
    print(f"Unfreeze Layers Count: {unfreeze_layers_count}")
    print(f"Unfreeze Layers: {unfreeze_layers}")
    print(f"Use Extra Perceptron: {use_extra_perceptron}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Use Dropout: {use_dropout}")
    print(f"Weight Decay: {weight_decay}")

    input_shape = model_name_input_shapes[model_name]
    
    # Load data
    train_loader, val_loader, train_dataset, val_dataset, label_map = data.load_data(input_shape, "~/gp/dataV5_augmented.csv")

    # Load model
    base_model = load(model_name, use_extra_perceptron, use_dropout)

    load_model_path = calc_model_path("/rds/user/omsst2/hpc-work/gp/results",
                                      model_name, learning_rate, unfreeze_layers_count,
                                      use_extra_perceptron, use_dropout, weight_decay)

    save_val_path = calc_model_path("/rds/user/omsst2/hpc-work/gp/results_v5_valoldmodel",
                                      model_name, learning_rate, unfreeze_layers_count,
                                      use_extra_perceptron, use_dropout, weight_decay)

    # Load weights
    model = serialisation.load_only_model(load_model_path, base_model)

    # Perform validation inference
    val_predictions = validate.validation_inference(model, val_dataset)

    # Save the results
    serialisation.save_val_preds(val_predictions, save_val_path)

if __name__ == "__main__":
    main()