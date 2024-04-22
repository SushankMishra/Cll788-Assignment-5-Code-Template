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
    def __init__(self, input_size, hidden_size,output_size,):
        super(LSTM, self).__init__()
        # To-do: Define the LSTM layer and a linear layer for required output size
        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, _ = self.LSTM(input)

        output = output.permute(1, 0, 2)
        
        final_output = torch.zeros(size = output.shape).to(input.device)
        for i in range(output.shape[0]):
            final_output[i, :, :] = self.fc(output[i, :, :])

        final_output.permute(1, 0, 2)

        return final_output

