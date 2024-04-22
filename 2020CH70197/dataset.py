"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
class Num10kDataset(Dataset):
    @staticmethod
    def text_to_dict(file_path):
        word_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                text, number_str = line.strip().split()
                # Convert the number string to an integer
                number = int(number_str)
                # Assign text as key and number as value in the dictionary
                word_dict[text] = number
        return word_dict
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # To-do: Load the images and labels from the folder
        files = os.listdir(root_dir)
        files = sorted(files)
        label_file = "labels.txt"
        label_dict = {}
        if label_file in files:
            label_dict = self.text_to_dict(os.path.join(root_dir,label_file))
        for filename in files:
            if filename.endswith('.jpg'):
                self.images.append(os.path.join(root_dir,filename))
                self.labels.append(label_dict[filename])
    

    def __len__(self):
        # To-do: Return the length of the dataset
        
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load the image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

        # To-do: Load the image and label at the given index, do the necessary processing and return them


class AlignCollate(object):
    """
        Used for collating and padding the images in the batch (labels being taken care by ConverterForCTC).
        Returns aligned image tensors and corresponding labels.
    """
    def __init__(self, imgH=32, imgW=100, input_channel=1):
        self.imgH = imgH
        self.imgW = imgW
        self.input_channel = input_channel
        self.transform = transforms.Compose([
            transforms.Resize((imgH, imgW)),  # Resize without maintaining aspect ratio
            transforms.ToTensor(),
            # Add normalization if needed
        ])
    def __call__(self, batch):
        images, labels = zip(*batch)
        aligned_images = []

        for image in images:

            preprocess = transforms.Compose([
                transforms.Pad(( int((self.imgW-image.size[0])/2) , int((self.imgH-image.size[1])/2) ), fill=255),
                transforms.Resize((self.imgH, self.imgW)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            image = preprocess(image)
            aligned_images.append(image)

        aligned_images = torch.stack(aligned_images, dim=0)
        # print(f"{aligned_images.shape = }")
        # aligned_images = torch.tensor(aligned_images)
        labels = torch.tensor(labels)

        return aligned_images, labels
        # To-do: Properly resize each image each in the batch to the same size
            # Make sure to maintain the aspect ratio of the image by using padding properly
            # Normalize the image and convert it to a tensor (If NOT already done in the Num10kDataset)
            # Concatenate the images in the batch to form a single tensor
        # Return the aligned image and corresponding labels