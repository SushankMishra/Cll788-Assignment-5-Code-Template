"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, output_size,hidden_size=256):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size)
        # Define a linear layer for the required output size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Initialize hidden state with zeros
        batch_size = input.size(0)
        hidden = self.init_hidden(batch_size)
        
        # Forward pass through the LSTM layer
        lstm_out, hidden = self.lstm(input, hidden)
        
        # Get the output of the last time step
        output = lstm_out[:, -1, :]
        
        # Forward pass through the linear layer
        output = self.linear(output)
        
        return output

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        # We need to specify the number of layers and directions (if using bidirectional LSTM)
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
