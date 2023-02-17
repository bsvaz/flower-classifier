import torch
from torchvision import transforms, datasets

def load_data(data_directory):
    
    '''Load data in the correct format to be used with the model
        Input:
        - data_directory: directory with the images'''
    
    # Transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    # directories
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    # Load the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_data.class_to_idx, train_loader, valid_loader, test_loader
