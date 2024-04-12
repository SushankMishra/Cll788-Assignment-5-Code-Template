"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch.nn as nn

class CNNModule(nn.Module):
    """ The CNN Model for feature extraction """
    def __init__(self, input_channel=1, output_channel=512):
        super(CNNModule, self).__init__()
        # To-do: Define the layers for the CNN Module

    def forward(self, input):
        # To-do: Implement the forward pass of the CNN Module