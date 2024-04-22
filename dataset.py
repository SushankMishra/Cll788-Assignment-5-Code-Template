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
        label_file = 'labels.txt'
        if label_file in files:
            label_dict = text_to_dict(os.path.join(root_dir,label_file))
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
        
    def __call__(self, batch):
        images, labels = zip(*batch)
        aligned_images = []

        for img in images:
            img = self.resize_image(img, self.imgH, self.imgW)
            aligned_images.append(img)

        aligned_images = torch.stack(aligned_images, dim=0)
        labels = torch.tensor(labels)

        return aligned_images, labels
        # To-do: Properly resize each image each in the batch to the same size
            # Make sure to maintain the aspect ratio of the image by using padding properly
            # Normalize the image and convert it to a tensor (If NOT already done in the Num10kDataset)
            # Concatenate the images in the batch to form a single tensor
        # Return the aligned image and corresponding labels
    def resize_image(self, image, imgH, imgW):
        """
        Resize image while maintaining aspect ratio
        """
        width, height = image.size
        aspect_ratio = width / height

        target_ratio = imgW / imgH

        if aspect_ratio > target_ratio:
            new_width = int(imgH * aspect_ratio)
            resized_image = image.resize((new_width, imgH), Image.BILINEAR)
            left_pad = (new_width - imgW) // 2
            right_pad = new_width - imgW - left_pad
            padded_image = Image.new('RGB', (new_width, imgH))
            padded_image.paste(resized_image, (left_pad, 0))
        else:
            new_height = int(imgW / aspect_ratio)
            resized_image = image.resize((imgW, new_height), Image.BILINEAR)
            top_pad = (new_height - imgH) // 2
            bottom_pad = new_height - imgH - top_pad
            padded_image = Image.new('RGB', (imgW, new_height))
            padded_image.paste(resized_image, (0, top_pad))

        return padded_image