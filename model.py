"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch.nn as nn
from rnn import LSTM
from cnn import CNNModule

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        """ CNN Module """
        self.CNNModule = CNNModule(args.input_channel, args.output_channel)
        self.CNN_output = args.output_channel

        """ RNN Module """
        self.RNNModule = LSTM(self.CNN_output, args.hidden_size, args.hidden_size)
        self.RNN_output = args.hidden_size

        """ Prediction Module """
        self.Prediction = nn.Linear(self.RNN_output, args.num_class)

    def forward(self, input):
        # Input Shape: BatchSize x 1 x imgH x imgW
        
        # To-do: Pass the input through the CNNModule and do necessary processing like permutation and reshaping
            # Expected output shape: BatchSize x OutputChannel x h x w
            # h and w depend upon the input image size and the architecture of the CNNModule

        # To-do: Pass the above processed output through the RNNModule
            # Expected output shape: BatchSize x TimeSteps x HiddenSize
            # TimeSteps depends upon the input image size and the architecture of the CNNModule
        
        # To-do: Pass the RNN output through the Prediction Layer
            # Expected output shape: BatchSize x TimeSteps x NumClass
        
        # Return the final output
