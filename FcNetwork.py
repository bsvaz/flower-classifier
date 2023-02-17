from torch import nn
import torch.nn.functional as F

# This file defines a fully connected network class to be used in the classifier part of a pre trained model 

class FcNetwork(nn.Module):
    
    '''Fully connected network used in the classifier part of the pre-trained model.'''

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) -1)])
                                   
        self.output = nn.Linear(hidden_layers[-1], output_size)
                                   
        self.dropout = nn.Dropout(p=drop_p)
                                   
    def forward(self, x):
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)