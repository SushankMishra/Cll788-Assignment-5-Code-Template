"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import os
from torch.utils.data import Dataset

class Num10kDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # To-do: Load the images and labels from the folder

    def __len__(self):
        # To-do: Return the length of the dataset
        return

    def __getitem__(self, idx):
        return
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
        
        # To-do: Properly resize each image each in the batch to the same size
            # Make sure to maintain the aspect ratio of the image by using padding properly
            # Normalize the image and convert it to a tensor (If NOT already done in the Num10kDataset)
            # Concatenate the images in the batch to form a single tensor
        # Return the aligned image and corresponding labels
