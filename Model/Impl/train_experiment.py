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

#! DROPOUT NOTES !#

# No effect without extra perceptron for: { AlexNet, InceptionV3, VGG16 }


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

epochs = 1 
patience = 5

#! END GLOBALS !#

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument('--model_name', type=str, required=True, 
                        help='Name of the model to be used or saved.')
    parser.add_argument('--unfreeze_layers_count', type=str, default="0", 
                        help='Number of layers to unfreeze, passed as a string (int).')
    parser.add_argument('--use_extra_perceptron', type=str, default="False", 
                        help='Whether to use an extra perceptron, passed as a string ("True" or "False").')
    parser.add_argument('--learning_rate', type=str, required=True, 
                        help='Learning rate, passed as a string representation of a float.')
    parser.add_argument('--dropout_p', type=str, default="None", 
                        help='Dropout probability, passed as a string representation of a float. None if no dropout.')
    parser.add_argument('--weight_decay', type=str, default="None", 
                        help='Weight decay for L2 regularisation, passed as a string representation of a float.')

    args = parser.parse_args()
    return args


def train(train_loader, val_loader, model_name, unfreeze_layers, use_extra_perceptron, learning_rate, dropout_p, weight_decay):

    if model_name == "resnet":
        return resnet.fine_tune_resnet(
            train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
            learning_rate=learning_rate, epochs=epochs, patience=patience)

    elif model_name == "plantnet":
        return plantnet.fine_tune_plantnet(
            train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
            learning_rate=learning_rate, epochs=epochs, patience=patience)

    elif model_name == "inaturalist":
        return inaturalist.fine_tune_inaturalist(
            train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
            learning_rate=learning_rate, epochs=epochs, patience=patience)

    elif model_name == "inceptionv3":
        return inceptionv3.fine_tune_inceptionv3(
            train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
            learning_rate=learning_rate, epochs=epochs, patience=patience)

    elif model_name == "alexnet":
        return alexnet.fine_tune_alexnet(
            train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
            learning_rate=learning_rate, epochs=epochs, patience=patience)

    elif model_name == "vgg16":
        return vgg16.fine_tune_vgg16(
            train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
            learning_rate=learning_rate, epochs=epochs, patience=patience)

    elif model_name == "densenet121":
        return densenet121.fine_tune_densenet121(
            train_loader, val_loader, use_extra_perceptron, unfreeze_layers, weight_decay, dropout_p,
            learning_rate=learning_rate, epochs=epochs, patience=patience)
    
    elif model_name == "simplecnn":
        return simplecnn.train_simplecnn(train_loader, val_loader, weight_decay, dropout_p,
                                         learning_rate=learning_rate, epochs=epochs, patience=patience)

    elif model_name == "deepverge":
        return deepverge.train_deepverge(train_loader, val_loader, weight_decay, dropout_p,
                                         learning_rate=learning_rate, epochs=epochs, patience=patience)

    else:
        raise ValueError("wah")


def main():
    args = parse_arguments()

    # Convert string arguments to appropriate types
    unfreeze_layers_count = int(args.unfreeze_layers_count)  # Convert string representation of list to actual list
    use_extra_perceptron = args.use_extra_perceptron.lower() == 'true'  # Convert string to boolean
    learning_rate = float(args.learning_rate)  # Convert string to float
    model_name = args.model_name  # String remains a string
    dropout_p =  None if args.dropout_p == "None" else float(args.dropout_p)
    weight_decay = None if args.weight_decay == "None" else float(args.weight_decay)

    if model_name not in model_name_input_shapes:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {list(model_name_input_shapes.keys())}")

    unfreeze_layers = []
    if model_name in model_finetune_layers:
        for i in range(unfreeze_layers_count):
            unfreeze_layers.append(model_finetune_layers[model_name][i])

    print(f"Model Name: {model_name}")
    print(f"Unfreeze Layers Count: {unfreeze_layers_count}")
    print(f"Unfreeze Layers: {unfreeze_layers}")
    print(f"Use Extra Perceptron: {use_extra_perceptron}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Dropout Probability: {dropout_p}")
    print(f"Weight Decay: {weight_decay}")

    input_shape = model_name_input_shapes[model_name]
    
    # Load data
    train_loader, val_loader, train_dataset, val_dataset, label_map = data.load_data(input_shape, "~/gp/dataV6_aug.csv")
    #train_loader, val_loader, train_dataset, val_dataset, label_map = data.load_data(input_shape, "~/gp/dataset-v4.csv")
    # train_loader, val_loader, train_dataset, val_dataset, label_map = data.load_data(input_shape, "~/gp/test_data.csv")

    # Train model
    model, train_stats, val_stats = train(train_loader, val_loader, model_name, unfreeze_layers, use_extra_perceptron, learning_rate, dropout_p, weight_decay)

    # Perform validation inference
    val_predictions = validate.validation_inference(model, val_dataset)

    # Save the results
    model_dir = f"/rds/user/omsst2/hpc-work/gp/results_v6_test/{model_name}_uf{unfreeze_layers_count}_p{use_extra_perceptron}_dp{dropout_p}_wd{weight_decay}_lr{learning_rate}"

    serialisation.save_model(model, train_stats, val_stats, val_predictions, model_dir)

if __name__ == "__main__":
    main()
