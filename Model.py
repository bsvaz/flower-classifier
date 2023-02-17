import time
import torch
from torchvision import models
from torch import nn, optim
from FcNetwork import FcNetwork
from workspace_utils import active_session
from loader import load_data

# This file defines a Model class that can be trained and save in a ".pth" format

class Model(nn.Module):

    def __init__(self, architecture, hidden_units, data_directory, device):
        
        super().__init__()
        
        self.architecture = architecture
        self.hidden_units = hidden_units
        self.data_directory = data_directory
        self.device = device
        
        class_to_idx, train_loader, valid_loader, test_loader = load_data(data_directory)

        self.class_to_idx = class_to_idx     
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.densenet121 = models.densenet121(pretrained=True)
        self.densenet161 = models.densenet161(pretrained=True)
        self.densenet201 = models.densenet201(pretrained=True)
        
        model_list = {'densenet121' : self.densenet121, 'densenet161' : self.densenet161, 
                        'densenet201' : self.densenet201}
        
        self.model = model_list[self.architecture]
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        fc_network = FcNetwork(self.model.classifier.in_features, 102, 
                                self.hidden_units)
        
        self.model.classifier = fc_network
        self.model.to(device)
    
    def forward(self, x):
        
        return self.model.forward(x)

    def train_model(self, epochs, criterion, lr):
        
        '''Method to train the model
        
           Inputs:
            -epochs: number of iteration used in the training process
            -criterion: the loss function used in the training process
            -lr: learning rate'''
        
        start = time.time()

        optimizer = optim.Adam(self.model.classifier.parameters(), lr = lr)
        
        with active_session():
            
            for e in range(epochs):
            
                train_loss = 0
                valid_loss = 0
                accuracy = 0
                
                for images, labels in self.train_loader:
            
                    images = images.to(self.device)
                    labels = labels.to(self.device)
            
                    log_ps = self.model(images)
                    loss = criterion(log_ps, labels)
                    train_loss += loss.item()
            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                self.model.eval()
                
                with torch.no_grad():
                    for images, labels in self.valid_loader:

                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        log_ps = self.model(images)
                        probs, top_class = log_ps.topk(1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += 100 * torch.mean(equals.type(torch.FloatTensor)).item()

                        valid_loss += criterion(log_ps, labels).item()
            
                train_loss = train_loss / len(self.train_loader)
                valid_loss = valid_loss / len(self.valid_loader)
                accuracy = accuracy / len(self.valid_loader)
            
                end = time.time()
            
                print('Epoch: {}, Train loss: {}, Valid loss: {}, Valid Accuracy: {}%,\
                    Time: {}s'.format(e + 1, train_loss, valid_loss, accuracy, end - start))
            
    def save_checkpoint(self, checkpoint):
        
        '''Method to save the model
            Input:
            - checkpoint: name of the file to be saved'''
        
        torch.save(self, checkpoint)
    

