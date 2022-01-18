#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
#Sources: From 4.12 mnist.py, 3.13 and #https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
import smdebug.pytorch as smd

#forum https://knowledge.udacity.com/questions/775194
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#resources:
#From forum questions #773197
#From lesson 3.13 fine-tune CNN
#From minst.py 4.12.hbo_deploy.py

def test(model, test_loader, loss_criterion, device):#, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    #hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)       
    total_acc = running_corrects.double() / len(test_loader)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    

def train(model, train_loader, validation_loader, loss_criterion, optimizer, device, epochs):#hook, epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    loss_counter=0
    best_loss=10000000
    #hook.set_mode(smd.modes.TRAIN) # set debugging hook
    image_dataset={'train':train_loader, 'valid':validation_loader}
    
    for epoch in range(epochs):
        logger.info(f"Epoch:{epoch}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                #hook.set_mode(smd.modes.TRAIN) # set debugging hook
            else:
                model.eval()
                #hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss,epoch_acc,best_loss))                                                 
                                                                                 
        if loss_counter==1:
            break
        if epoch==0:
            break
    return model


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model



def create_data_loaders(data, batch_size):
    
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
                                                            
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    #Get data from s3
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    logger.info("Hyperparameters: epoch: {}, lr: {}, batch size: {}".format(
                    args.epochs, args.lr, args.batch_size)
    logger.info(f'Data Path: {args.data}')
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    #From lesson 3.13 fine-tune CNN
    model=net()
                
    #if cuda...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    print(f"Running on Device {device}")
                   
    #hook = smd.Hook.create_from_json_file()
    #hook.register_module(model)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    #hook.register_loss(loss_criterion)
                
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)        
    logger.info("Training...")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, args.epochs)#, hook, )

    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing...")
    test(model, test_loader, loss_criterion, device)#, hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for testing (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args = parser.parse_args()
    main(args)
    
    
    
    
    """
#funca ok sin el agregado de collection_configs en el jupyter nb
#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
#Sources: From 4.12 mnist.py, 3.13 and #https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed

import argparse
import logging
import os
import sys
import json


#forum https://knowledge.udacity.com/questions/775194
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from smdebug import modes
from smdebug.pytorch import get_hook


def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model...")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")

def train(model, train_loader, test_loader, batch_size, epochs, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(criterion)
        hook.register_module(model) #mentor suggestion
        
    for epoch in range(1, epochs + 1):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Training Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, criterion)
    

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

def _get_train_data_loader(batch_size, training_dir):
    logger.info("Train data loader")
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    trainset = torchvision.datasets.ImageFolder(root=training_dir,
            transform=transform_train)
    return torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True
        )


def _get_test_data_loader(batch_size, test_dir):
    logger.info("Test data loader")
    testing_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testset = torchvision.datasets.ImageFolder(root=test_dir, 
            transform=testing_transform)
    return torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,
            shuffle=False
        )

def main(args):
    logger.info("Hyperparameters: epoch: {}, lr: {}, batch size: {}, momentum: {}".format(
                    args.epochs, args.lr, args.batch_size, args.momentum)
    )
    '''
    TODO: Initialize a model by calling the net function
    '''
    #From lesson 3.13 fine-tune CNN
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''    
    #train_loader, test_loader = create_data_loaders(args.train_dir, args.test_dir, args.batch_size)
    train_loader = _get_train_data_loader(args.batch_size, args.train_dir)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test_dir)
    #test_loader = _get_test_data_loader(args.batch_size, args.test_dir)
    
    train(model, train_loader, test_loader, args.batch_size, args.epochs, criterion, optimizer)

    '''
    TODO: Test the model to see its accuracy
    '''
    #the test is performed into the train function 
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pth')
    #torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Saving model: {}, to path: {}".format(model, path))
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for testing (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    main(parser.parse_args())
"""
