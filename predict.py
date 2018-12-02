import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from PIL import Image
import json

gpu_available = torch.cuda.is_available

def process_image(image_path):
    image_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])
    image = Image.open(image_path)
    image = image_transform(image)
    np_image = np.array(image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image

def load_checkpoint(chkpt):
    checkpoint = torch.load(chkpt)
    state_dict = checkpoint['state_dict']
    #print(state_dict)
    mm = checkpoint['model']
    mm.load_state_dict(state_dict)
    
    return mm

def predict(args, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
     
    image = process_image(args.image_uri)
    image_tensor = torch.FloatTensor([image])
    
    idx_to_class = torch.load(args.checkpoint)['class_to_idx']
    idx_to_class = {i: j for j, i in idx_to_class.items()}
    print('idx_to_class', idx_to_class)
    
    if args.gpu and gpu_available:
        image_tensor = image_tensor.cuda()
        model = model.cuda()
    else:
        image_tensor = image_tensor.cpu()
        model = model.cpu()
    
    output = model.forward(image_tensor)
    ps = torch.exp(output).data[0]
    class_index = ps.topk(args.topk)
    
    classes = [idx_to_class[i]
               for i in class_index[1].cpu().numpy()]
    probabilities = class_index[0].cpu().numpy()

    print('Probabilities: ', probabilities)

    return probabilities, classes

def main():
    parser = argparse.ArgumentParser(description='image classifier predictor')
    parser.add_argument('--gpu', type=bool, default=False, help='use gpu if specified')
    parser.add_argument('--image_uri', type=str, help='path to image u want to predict')
    parser.add_argument('--topk', type=int, default=5, help='topk to return')
    parser.add_argument('--checkpoint', type=str, default='model_checkpoint.pth', help='saved tained model checkpoint')
    parser.add_argument('--json_path', type=str, default='cat_to_name.json', help='path to json')
    args = parser.parse_args()
    
    with open(args.json_path, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load_checkpoint(args.checkpoint)
    probabilities, classes = predict(args, model)
    
    print('classes: ', classes)
    [print(cat_to_name[x]) for x in classes]
    print('probabilities: ', probabilities)
    
if __name__ == "__main__":
    main()
    
    