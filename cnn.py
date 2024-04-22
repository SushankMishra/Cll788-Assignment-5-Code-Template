"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch.nn as nn
import torchvision.models as models

class CNNModule(nn.Module):
    """ The CNN Model for feature extraction """
    def __init__(self, input_channel=1, output_channel=512):
        super(CNNModule, self).__init__()
        # Load the pretrained ResNet model
        resnet = models.resnet18(pretrained=True)
        # Remove the fully connected layers at the end
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Modify the first convolutional layer to accept input_channel
        if input_channel != 3:
            self.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Define the output_channel
        self.output_channel = output_channel

    def forward(self, input):
        # Forward pass through the ResNet feature extractor
        features = self.features(input)
        # Return the extracted features
        return features