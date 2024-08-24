from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import inaturalist
import serialisation

# images_path = "/home/ubuntu/all_images"
images_path = "/app/all_images"
# model_path = "/home/ubuntu/models/inaturalist_uf5_pFalse_lr0.0001"
model_path = "/app/model"
use_extra_perceptron = False
labels = ['Grass', 'Clover', 'Soil', 'Dung']

def get_transform(img_dimensions=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_dimensions),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_and_transform_image(image_path):
    # Load and transform image to tensor suitable for model input
    image = Image.open(image_path)
    transform = get_transform()
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

def to_machine_image_path(path):
    file_name = os.path.basename(path)
    file_name = file_name.replace('%20', ' ')
    actual_path = os.path.join(images_path, file_name)
    return actual_path

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.2")

        self.model = inaturalist.load(use_extra_perceptron, weights=False)
        self.model = serialisation.load_only_model(model_path, self.model)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        df = [load_and_transform_image(to_machine_image_path(task['data']['image'])) for task in tasks]

        predictions = inf(self.model, df)

        print(f"Predictions: {predictions}")

        preds = []
        for i, pred in enumerate(predictions):
            result_labels = [labels[j] for j, score in enumerate(pred) if score > 0.5]
            score = 2 * float(min(abs(float(x) - 0.5) for x in pred))
            preds.append({
                "model_version": self.get("model_version"),
                "score": score,
                "result": [
                    {
                        "id": f"result_{i}",
                        "type": "choices",
                        "from_name": "choice",
                        "to_name": "image",
                        "score": score,
                        "value": {
                            "choices": result_labels
                        }
                    }
                ]
            })

        return ModelResponse(predictions=preds)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
