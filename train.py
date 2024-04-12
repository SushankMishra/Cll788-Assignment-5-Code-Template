"""
Assignment 5: CRNN For Text Recognition

Course Coordinator: Dr. Manojkumar Ramteke
Teaching Assistant: Abdur Rahman

This code is for educational purposes only. Unauthorized copying or distribution without the consent of the course coordinator is prohibited.
Copyright © 2024. All rights reserved.
"""

import os
import time
import argparse

import torch
import time
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt

from model import Model
from test import validation
from dataset import  AlignCollate, Num10kDataset
from utils import ConverterForCTC

def train(args, device):
    args.device = device
    print('\n'+'-' * 80)
    print('Device : {}'.format(device))
    print('-' * 80 + '\n')
    
    align_collate_function = AlignCollate(imgH=args.imgH, imgW=args.imgW, input_channel=args.input_channel)
    train_dataset = Num10kDataset(args.train_data)
    print("Loaded Train Dataset, Length: ", len(train_dataset))
    valid_dataset = Num10kDataset(args.valid_data)
    print("Loaded Validation Dataset, Length: ", len(valid_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,
        collate_fn=align_collate_function)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=True,
        collate_fn=align_collate_function)
    
    # To-do: Create an instance of the ConverterForCTC
        # Use the args.character to initialize the ConverterForCTC
        # Set the number of classes in the args using the length of the character in the converter
    
    # To-do: Create an instance of the Model
        # Use the args to initialize the Model
        # Print the number of trainable parameters in the model
        # Move the model to the device
    
    # To-do: Define the loss function using torch.nn.CTCLoss
        # Set the blank label to 0
        # Set the reduction to 'mean'

    # To-do: Define the optimizer
    
    init_time = time.time()
    # To-do: Write the training loop
        # Iterate over the training data
        # Pass the image to the model
        # Encode the labels using the converter
        # Calculate the loss using the criterion
        # Backpropagate the loss and model parameters
        # Evaluate the model on the validation dataset
        # Print the training and validation loss and accuracy
        # Save the best model based on the validation loss
        # Plot the loss curve (Both training and validation loss)
    
    end_time = time.time()
    print("Total time taken for training: " + str(end_time-init_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='saved_models', help='path to save model')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--batch_max_length', type=int, default=0, help='Maximum label length') # DECIDE APPROPRIATELY BY OBSERVING THE DATASET
    parser.add_argument('--imgH', type=int, default=0, help='the height of the input image') # DECIDE APPROPRIATELY BY OBSERVING THE DATASET
    parser.add_argument('--imgW', type=int, default=0, help='the width of the input image') # DECIDE APPROPRIATELY BY OBSERVING THE DATASET
    
    """ Model Architecture - DO NOT CHANGE """
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel for CNN')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel for CNN')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    """ vocab / character number configuration """
    args.character = "0123456789"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)
    
    train(args, device)