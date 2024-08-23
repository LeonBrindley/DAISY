import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import ast
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
        def __init__(self, dataframe, img_dir, transform=None):
            self.dataframe = dataframe
            self.img_dir = img_dir
            self.transform = transform
            self.label_map = {'Grass': 0, 'Clover': 1, 'Soil': 2, 'Dung': 3} # NB 4 classes
            self.num_classes = len(self.label_map)
            self.class_names = list(self.label_map.keys())
            self.image_paths = [os.path.join(self.img_dir, img_name) for img_name in self.dataframe.iloc[:, 6]]

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_name = self.image_paths[idx]
            image = Image.open(img_name).convert("RGB")

            # Extract and process the label string
            label_str = self.dataframe.iloc[idx, 9]
            label_list = ast.literal_eval(label_str)  # Safely evaluate the string to a list
            label_indices = [self.label_map[label] for label in label_list]  # Map to indices
            label_tensor = torch.tensor(label_indices, dtype=torch.long)

            label_onehot = nn.functional.one_hot(label_tensor, num_classes=self.num_classes)
            label_onehot = label_onehot.max(dim=0)[0].float()

            if self.transform:
                image = self.transform(image)

            return image, label_onehot

        def get_image_path(self, idx):
            return self.image_paths[idx]

        def get_labels(self, idx):
            return self.dataframe.iloc[idx, 9]

def get_transform(img_dimensions=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_dimensions),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def onehot_to_labels(onehot_label_tensor, class_names):
    """
    Convert a one-hot encoded label tensor to a list of label names.
    """
    # Ensure the input is a tensor
    if not isinstance(onehot_label_tensor, torch.Tensor):
        raise TypeError("Input should be a torch.Tensor")

    # Convert the one-hot tensor to a list of indices where value is 1
    active_indices = torch.where(onehot_label_tensor > 0.5)[0].tolist() # threshold of 0.5

    # Map indices to label names
    labels = [class_names[idx] for idx in active_indices]

    return labels


def load_data(img_dimensions=(229,229), csv_path = "~/gp/dataset-v4.csv"):
    main_df = pd.read_csv(csv_path)
    main_df['image_name'] = main_df['image']
    main_df['label'] = main_df['labels']
    main_df_noNaN = main_df.dropna(subset=['label'])
    print(f"{len(main_df) - len(main_df_noNaN)} images with no labels removed")

    transform = get_transform(img_dimensions)

    img_dir = '/rds/user/omsst2/hpc-work/gp/data/content/content/flat_split'

    train_df = main_df[main_df['subset'] == 'Train']
    val_df = main_df[main_df['subset'] == 'Val']

    train_dataset = CustomDataset(dataframe=train_df, img_dir=img_dir, transform=transform)
    val_dataset = CustomDataset(dataframe=val_df, img_dir=img_dir, transform=transform)

    label_map = train_dataset.label_map

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f'Train set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')

    return train_loader, val_loader, train_dataset, val_dataset, label_map


