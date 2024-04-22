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
        self.LinearTransform = nn.Linear(32 * args.output_channel , args.output_channel)
    def forward(self, input):
        # Input Shape: BatchSize x 1 x imgH x imgW
        print("Shape of input sensor", input.shape)
        # Pass the input through the CNNModule and do necessary processing like permutation and reshaping
        cnn_output = self.CNNModule(input)
        batch_size, channels, imgH, imgW = cnn_output.shape
        cnn_output = cnn_output.permute(0, 3, 1, 2)  # Permute to BatchSize x h x w x OutputChannel
        cnn_output = cnn_output.view(batch_size, imgW, channels * imgH)  # Reshape to BatchSize x (h*w) x OutputChannel
        
        # Pass the above processed output through the RNNModule
        cnn_output = self.LinearTransform(cnn_output)
        rnn_output = self.RNNModule(cnn_output)
        
        # Pass the RNN output through the Prediction Layer
        prediction_output = self.Prediction(rnn_output)
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
        return prediction_output
