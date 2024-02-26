import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

classes = {'CNV': 0, 'DME': 1, 'Drusen': 2, 'Normal': 3}


class train_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_path = os.path.join(data_path, 'train')
        self.images = []
        self.labels = []

        for i in classes:
            images_path = os.path.join(self.data_path, i)
            label = classes[i]
            for j in os.listdir(images_path):
                image_path = os.path.join(images_path, j)
                image = Image.open(image_path)
                image = image.convert("RGB")
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.images)


class val_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_path = os.path.join(data_path, 'val')
        self.images = []
        self.labels = []

        for i in classes:
            images_path = os.path.join(self.data_path, i)
            label = classes[i]
            for j in os.listdir(images_path):
                image_path = os.path.join(images_path, j)
                image = Image.open(image_path)
                image = image.convert("RGB")
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.images)


class test_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_path = os.path.join(data_path, 'test')
        self.images = []
        self.labels = []

        for i in classes:
            images_path = os.path.join(self.data_path, i)
            label = classes[i]
            for j in os.listdir(images_path):
                image_path = os.path.join(images_path, j)
                image = Image.open(image_path)
                image = image.convert("RGB")
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.images)
