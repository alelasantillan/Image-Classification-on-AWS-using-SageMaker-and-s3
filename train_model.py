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

#resources:
#From forum questions #773197
#From lesson 3.13 fine-tune CNN
#From minst.py 4.12.hbo_deploy.py



#def test(model, test_loader, criterion):
def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #From lesson 3.13 fine-tune CNN
    #a revisar
    print("Testing...")
    model.eval()
    running_loss=0
    running_corrects=0
    
    loss_criterion = nn.CrossEntropyLoss()
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    #pass
    



#def train(model, train_loader, criterion, optimizer):
def train(args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #From lesson 3.13 fine-tune CNN
    logger.info("Hyper_parameters: epoch: {}, lr: {}, batch size: {}, momentum: {}".format(
                    args.epochs, args.lr, args.batch_size, args.momentum)
    )
    #train_loader, test_loader = create_data_loaders(data, batch_size)
    
    #train_loader, test_loader = create_data_loaders(args.train_dir,args.test_dir, args.batch_size )
    train_loader, test_loader = create_data_loaders(args)
    
    #train_loader = _get_train_data_loader(args.batch_size, args.train_dir)
    #test_loader = _get_test_data_loader(args.test_batch_size, args.test_dir)

    
    model = net()
    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(loss_optim)

    for epoch in range(1, args.epochs + 1):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_optim(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    return model
    

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    #From lesson 3.13 fine-tune CNN
    #we fix all layers except the linear one and we change 10 (digits) for 133 features in 
    #dog breeds
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    #model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model
    #pass


#combine _get_train_data_loader and _get_test_data_loader into one function
#def create_data_loaders(train_dir, test_dir, batch_size):
def create_data_loaders(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    batch_size = args.batch_size
    #def _get_train_data_loader(batch_size, train_dir):
    #train_dir = os.path.join(data, "train")
    #test_dir = os.path.join(data, "test")
    #train_dir=train_dir
    #test_dir=test_dir
    logger.info("Train path:",train_dir)
    logger.info("Test path:",test_dir)
    logger.info("Training...")
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    trainset = torchvision.datasets.ImageFolder(root=train_dir,
            transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True
        )


    #def _get_test_data_loader(batch_size, test_dir):
    logger.info("Test data loader")
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    testset = torchvision.datasets.ImageFolder(root=test_dir, 
            transform=transform_test)
    testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,
            shuffle=False
        )

    return trainloader, testloader

#https://knowledge.udacity.com/questions/760362
def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #From lesson 3.13 fine-tune CNN
    #model=create_model(), now called net()
    #model=model.to(device)
    model=net()
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    #loss_criterion = None
    #optimizer = None
    #From lesson 3.13 fine-tune CNN
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #try then using AdamW
    #optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, eps= args.eps, weight_decay = args.weight_decay)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    #train_loader, test_loader = create_data_loaders(args.train_dir, args.test_dir, args.batch_size) #yo
    #train_loader, test_loader = create_data_loaders(train_dir=args.train_dir, test_dir=args.test_dir, batch_size=args.batch_size) #yo
    train_loader, test_loader = create_data_loaders(args) #yo
    #model=train(model, train_loader, criterion, optimizer)
    model=train(parser.parse_args())
    
    '''
    TODO: Test the model to see its accuracy
    '''
    #test(model, test_loader, criterion)
    test(model, test_loader)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pth')
    #torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    print("Saving model: ",model," to path: ", path)
    torch.save(model.state_dict(), path)
    
    
    

if __name__=='__main__':
    #source https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#id8
    #mentor answer
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    #From minst.py 4.12.hbo_deploy.py, but using pretrained Resnet18 instead of defining our own model
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    
    #parser.add_argument(
    #    "--test_batch_size",
    #    type=int,
    #    default=1024,
    #    metavar="N",
    #    help="input batch size for testing (default: 64)",
    #)
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    #a revisar los environment vars generated by sagemaker in train_and_deploy.ipynb to get training data, model_dir and output_dir in S3. See output of the cell that contains estimator.fit({
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]) 
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    #parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR']) 
    #train(parser.parse_args())
    
    args=parser.parse_args()
    #args, _ =parser.parse_known_args()
    
    
    main(args)
    

