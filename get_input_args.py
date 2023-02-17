import argparse 
import torch
from torch import nn

# This file contains functions that permit interaction with the code through the command line

def get_input_train():
    
    parser = argparse.ArgumentParser(description='Get arguments to train the model')
    
    # Model class arguments
    parser.add_argument('-arch', '--architecture', default='densenet201')
    parser.add_argument('-hu', '--hidden_units', default=[512, 256])
    parser.add_argument('-dev', '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train_model method arguments
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--criterion', default=nn.NLLLoss())
    parser.add_argument('-lr', '--learning_rate', default=0.003)
    parser.add_argument('--data_directory', default='flowers', help='Directory where images are taken to be trained')
    parser.add_argument('--save_dir', default='save_directory', help='Directory where the model will be saved')
    parser.add_argument('checkpoint', help='Name of the file where the model will be saved')

    return parser.parse_args()

def get_input_predict():

    parser = argparse.ArgumentParser(description='Get arguments to predict the \
                                                  specie of the flower')
    parser.add_argument('-i', '--image_path')
    parser.add_argument('-c', '--checkpoint', default='save_directory/checkpoint.pth')
    parser.add_argument('-k', '--topk', default=5)
    parser.add_argument('-d', '--device', default='cuda' if torch.cuda.is_available()
                                                 else 'cpu')
    parser.add_argument('-n', '--category_names', default='cat_to_name.json')

    return parser.parse_args()
    