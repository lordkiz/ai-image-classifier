import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import json
from collections import OrderedDict

gpu_available = torch.cuda.is_available

def data_prep(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=102, shuffle=True)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=102, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=102, shuffle=True)
    
    return dataloaders, image_datasets

def model_train(args, model, criterion, optimizer, epochs, steps, print_every):
    if gpu_available and args.gpu:
        model.to('cuda')
    dataloaders, image_datasets = data_prep(args)
    train_dataloaders = dataloaders['train']
    valid_dataloaders = dataloaders['valid']
    
    for e in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in iter(train_dataloaders):
            steps += 1
            
            if gpu_available and args.gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                running_valid_loss = 0
                valid_total = 0
                valid_correct = 0
                model.eval()
                for ii, (inputs, labels) in enumerate(valid_dataloaders):
                    if gpu_available and args.gpu:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    output = model.forward(inputs)
                    _, predicted = torch.max(output.data, 1)
                    running_valid_loss += criterion(output, labels).item()
                    valid_total += labels.size(0)
                    valid_correct += (predicted == labels).sum().item()

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train Loss: {:.4f}".format(running_loss/print_every),
                      "Train Accuracy: %d %%" % (100 * correct / total)
                     )
                print('-------------------------------------------')
                print("Valid Loss: {:.4f} ...".format(running_valid_loss/print_every),
                      "Valid Accuracy: %d %%" % (100 * valid_correct / valid_total)
                     )
                print('********************************************')
                print()

                running_loss = 0
    return model



def train_wrapper(args):
    if args.arch == 'vgg':
        model = models.vgg19(pretrained=True)
        input_features = model.classifier[0].in_features
        output_features = model.classifier[0].out_features
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features
        output_features = model.classifier.out_features
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, output_features)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('hidden units', nn.Linear(output_features, args.hidden_units)),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    if args.gpu and gpu_available:  #use gpu
        model = model.cuda()
        print('Using gpu')
    else:
        print('Using cpu')
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.lr)
    
    model = model_train(args, model, criterion, optimizer, epochs=args.epochs, steps=0, print_every=40)
    
    all_dataloaders, all_datasets = data_prep(args)
    print('datasets', all_datasets['train'])
    model.class_to_idx = all_datasets['train'].class_to_idx
    checkpoint = { 
        'epochs': args.epochs,
        'batch_size': 102,
        'classifier': classifier,
        'model': model,
        'optimizer': optimizer,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'criterion': criterion
        }
    torch.save(checkpoint, 'model_checkpoint.pth')
    
def main():
    parser = argparse.ArgumentParser(description='Image classification transfer training')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if specified')
    parser.add_argument('--epochs', type=int, default=10, help='epochs to use')
    parser.add_argument('--hidden_units', type=int, default=100, help='number of hidden units')
    parser.add_argument('--arch', type=str, default='vgg', help='desired architecture: either densenet or vgg19')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--saved_model' , type=str, default='model_checkpoint.pt', help='path to the saved model')
    args = parser.parse_args()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    train_wrapper(args)
    
if __name__ == "__main__":
    main()