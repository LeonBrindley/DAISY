import inference_service.omar_dir.preprocess as preprocess
import inference_service.omar_dir.data as data
import inference_service.omar_dir.resnet as resnet
import inference_service.omar_dir.plantnet as plantnet
import inference_service.omar_dir.inaturalist as inaturalist
import inference_service.omar_dir.alexnet as alexnet
import inference_service.omar_dir.vgg16 as vgg16
import inference_service.omar_dir.densenet121 as densenet121
import inference_service.omar_dir.serialisation as serialisation
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import re

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
    elif "alexnet" in model_path:
        base_model = alexnet.load(use_extra_perceptron, dropout_p)
    elif "vgg16" in model_path:
        base_model = vgg16.load(use_extra_perceptron, dropout_p)
    elif "densenet121" in model_path:
        base_model = densenet121.load(use_extra_perceptron, dropout_p)
    else:
        raise ValueError("Unsupported model type in model_path")

    # model = serialisation.load_only_model(model_path, base_model)
    return base_model

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

def remove_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform inference with a given model on a given image.')

    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Path to model directory.')

    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to image.')

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


def predict(model, image_path, work_dir):

    # Split image
    unique_id, fragment_paths = preprocess.preprocess_and_split_image(image_path, work_dir)

    # Load and transform image fragments
    df = [load_and_transform_image(path) for path in fragment_paths]


    # print(fragment_paths)
    # Perform inference
    predictions = inf(model, df)

    # To write down the prediction but just for testing now

    # with open('/tmp/work/predictions.txt', 'w') as file:
    #     for i, pred in enumerate(predictions):
    #         file.write(f'{fragment_paths[i]}={pred}\n')
    # Cleanup: Delete each fragment




    for path in fragment_paths:
        remove_file(path)
    print(f"Deleted fragments at {work_dir}")

    # print(f"Predictions for {image_path}: {predictions}")
    # Merge the predictions
    merged_predictions = merge_predictions(predictions)
    print(f"Merged predictions for {image_path}: {merged_predictions}")

    return fragment_paths, predictions, merged_predictions

# req: pip3 install pillow-heif
def main():
    setup()

    args = parse_arguments()

    # Convert string arguments to appropriate types
    model_dir = args.model_dir
    image_path = args.image_path
    work_dir = args.work_dir

    print(f"Model Directory: {model_dir}")
    print(f"Image Path: {image_path}")
    print(f"Work Directory: {work_dir}")

    # Load model
    model = load_model(model_dir)

    # Split image
    unique_id, fragment_paths = preprocess.preprocess_and_split_image(image_path, work_dir)

    # Load and transform image fragments
    df = [load_and_transform_image(path) for path in fragment_paths]

    # Perform inference
    predictions = inf(model, df)

    # Cleanup: Delete each fragment
    for path in fragment_paths:
        remove_file(path)
    print(f"Deleted fragments at {work_dir}")

    print(f"Predictions for {image_path}: {predictions}")

    # Merge the predictions
    merged_predictions = merge_predictions(predictions)
    print(f"Merged predictions for {image_path}: {merged_predictions}")


if __name__ == "__main__":
    main()
