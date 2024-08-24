import preprocess
import data
import resnet
import plantnet
import inaturalist
import alexnet
import vgg16
import densenet121
import ensemble
import validate
import inceptionv3
import simplecnn
import deepverge
import serialisation
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import re
import glob
import dropout

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



def load_and_transform_image(image_path):
    # Load and transform image to tensor suitable for model input
    image = Image.open(image_path)
    transform = data.get_transform()
    image_tensor = transform(image)
    return image_tensor

def inf(model, df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    predictions = []

    with torch.no_grad():
        for data in tqdm(df):
            inputs = data.to(device)
            inputs = inputs.unsqueeze(0)  # Add batch dimension

            output = model(inputs)
            predictions.append(output.cpu().detach().numpy())

    predictions = np.squeeze(np.array(predictions), axis=1)
    return predictions

def merge_predictions(predictions):
    # Take the maximum value for each value across all the predictions
    merged_predictions = np.max(predictions, axis=0)
    return merged_predictions

def dist_predictions(predictions):
    mean_predictions = np.mean(predictions, axis=0)
    dev_predictions = np.std(predictions, axis=0)
    return mean_predictions, dev_predictions

def remove_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform inference with a given model on a given image.')

    parser.add_argument('--image_dir', type=str, required=True, 
                        help='Path to images dir.')

    parser.add_argument('--work_dir', type=str, required=True, 
                        help='Path to work directory.')

    args = parser.parse_args()
    return args

def register_heif_opener():
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()

def setup():
    register_heif_opener()

# req: pip3 install pillow-heif
def main():
    setup()

    args = parse_arguments()

    # Convert string arguments to appropriate types
    image_dir = args.image_dir
    work_dir = args.work_dir

    print(f"Image Directory: {image_dir}")
    print(f"Work Directory: {work_dir}")


    # Function to find all image files in the directory
    def find_image_files(directory):
        image_extensions = ('*.jpg', '*.jpeg', '*.heic')
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(directory, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        return image_files

    # Find all image files in the directory
    image_paths = find_image_files(image_dir)
    image_paths_str = '\n'.join(image_paths)
    print(f"Image Paths: {image_paths_str}")

    models_paths = "/rds/user/omsst2/hpc-work/gp/results_v6_grid/inaturalist_uf5_pFalse_dp0.1_wd1e-05_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/densenet121_uf7_pTrue_dp0.3_wd1e-05_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/inceptionv3_uf10_pTrue_dp0.3_wd0.01_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/alexnet_uf7_pTrue_dpNone_wd0.0015_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_gpu/vgg16_uf6_pTrue_dp0.3_wdNone_lr1e-05,/rds/user/omsst2/hpc-work/gp/results_v6_grid/deepverge_uf0_pFalse_dp0.5_wdNone_lr0.0001,/rds/user/omsst2/hpc-work/gp/results_v6_grid/simplecnn_uf0_pFalse_dp0.1_wd0.01_lr0.0001".split(',')

    models_paths = models_paths[:5]
    model_names = [model_path_to_model_name(model_path) for model_path in models_paths]

    print(f"Model names:\n{model_names}")
    print(f"Model paths:\n{models_paths}")

    # Load base models
    models = [load_model(model_path) for model_path in models_paths]
    input_sizes = [model_name_input_shapes[model_name] for model_name in model_names]


    base_ensemble_model = ensemble.LargeEnsembleModel(models, 4, input_sizes)

    dropout_p = 0.3

    if dropout_p is not None:
        base_ensemble_model = dropout.add_dropout(base_ensemble_model, "oe_fc", dropout_p)
        base_ensemble_model = dropout.add_dropout(base_ensemble_model, "oe_classifier", dropout_p)


    ensemble_model = serialisation.load_only_model("/home/omsst2/gp/best_models/best_ensemble/f1/ensemble_nm5_s2_dp0.3_wd0.01_lr0.001", base_ensemble_model)

    for image_path in image_paths:
        # Split image
        unique_id, fragment_paths = preprocess.preprocess_and_split_image(image_path, work_dir)

        # Load and transform image fragments
        df = [load_and_transform_image(path) for path in fragment_paths]

        # Perform inference
        predictions = inf(ensemble_model, df)

        # Cleanup: Delete each fragment
        for path in fragment_paths:
            remove_file(path)
        print(f"Deleted fragments at {work_dir}")

        #print(f"Predictions for {image_path}: {predictions}")

        # Merge the predictions
        merged_predictions = merge_predictions(predictions)
        print(f"Merged predictions for {image_path}: {merged_predictions}")

        mean_predictions, dev_predictions = dist_predictions(predictions)
        print(f"Mean predictions for {image_path}: {mean_predictions}")
        print(f"Dev predictions for {image_path}: {dev_predictions}")
        
if __name__ == "__main__":
    main()
