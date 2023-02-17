
from Model import Model
from get_input_args import get_input_train, get_input_predict

# File that will be called in terminal to train a model

def main():
    
    in_args = get_input_train()
    
    model = Model(in_args.architecture, in_args.hidden_units, in_args.data_directory, in_args.device)
    print('Object created')
    model.train_model(in_args.epochs, in_args.criterion, in_args.learning_rate)
    print('Model trained')
    model.save_checkpoint(in_args.save_dir + '/' + in_args.checkpoint)
    print('Checkpoint saved')
    

if __name__ == '__main__':
    main()