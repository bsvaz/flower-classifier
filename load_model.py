import torch
from Model import Model

def load_checkpoint(checkpoint_path, device):
    
    '''Load the checkpoint file into a model
        Inputs:
        - checkpoint_path: path of the checkpoint
        - device: if the model will be loaded on cpu or cuda'''
    
    checkpoint = torch.load(checkpoint_path)
    model = Model(checkpoint.architecture, checkpoint.hidden_units, checkpoint.data_directory, checkpoint.device)
    model.load_state_dict(checkpoint.state_dict())
    model.to(device)
    
    return model
