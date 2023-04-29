# Flower Species Classifier

This repository contains the final project for the AI Programming with Python course from Udacity. The project utilizes PyTorch to train and evaluate a deep learning model for classifying flower species using the command line. Due to GitHub's file size limitations, the training and testing images, as well as the trained model, are not included in this repository, but you can find the dataset <a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html" target="_blank">here</a>. Please note that running this project can be computationally expensive, and therefore, detailed instructions on how to execute the code have not been provided in this README file.

## Table of Contents

- Overview
- File Structure

## Overview

The goal of this project is to build a deep learning model capable of classifying flower species using images. The model can be trained using the command line and saves the trained model in a '.pth' format. The trained model can then be used to classify new images of flowers.

## File Structure

The repository is structured as follows:

- Fc.Network.py: Defines the fully connected network class to be used as a classifier in the Model class.
- Model.py: Contains the Model class that can be trained and saved in a .pth format.
- Image_Classifier_Project.ipynb: A Jupyter notebook with an outline to prepare for the command line part of the project.
- LICENSE: The license file for this project.
- cat_to_name.json: A JSON file containing the mapping between the numerical class and the name of the flower species.
- get_input_args.py: Responsible for getting the command line arguments using the argparse module.
- load_model.py: Contains the load_checkpoint function responsible for loading the checkpoint file into a model.
- loader.py: Contains the load_data function, responsible for loading the data in the correct format to be used in the model.
- train.py: Responsible for training the model.
- workspace_utils.py: Contains utility functions to prevent the server from crashing.
