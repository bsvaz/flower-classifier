from get_input_args import get_input_predict
from predict_functions import predict
from Model import Model 

# File that will be called in the terminal

if __name__ == '__main__':
    
    in_args = get_input_predict()
    predict(in_args.image_path, in_args.checkpoint, in_args.topk, in_args.device, in_args.category_names)
