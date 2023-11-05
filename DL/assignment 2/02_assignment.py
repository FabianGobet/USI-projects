'''
Assignment 2
Student: Fabian Gobet
'''
# *** Packges ***
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# *** Functions ***
def histo(dataset, set_name):
    labels = dataset.classes
    counts = [0 for _ in labels]
    dic = {c:None for c in labels}

    for img, lbl in dataset:
        if dic[labels[lbl]] is None:
            dic[labels[lbl]] = img
        counts[lbl] = counts[lbl]+1 

    fig = plt.figure(figsize=(10,5))
    fig.suptitle(set_name+" class image sample")
    for i in range(1,11):
        ax = fig.add_subplot(2,5,i)
        ax.imshow(dic[labels[i-1]])
        ax.axis('off')
        ax.set_title(labels[i-1])
    plt.show()

    fig, ax = plt.subplots(figsize=(10,5))
    fig.suptitle(set_name+" frequency per class")
    plt.bar(dataset.classes, counts)
    plt.xticks(fontsize=8)
    plt.ylabel("frequency")
    plt.show()
    

if __name__ == "__main__":
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)


    # 1.1.1
    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    #histo(dataset_train, "Cifar10 train set")
    #histo(dataset_test, "Cifar10 test set")
    
    # 1.1.2
    '''
    sample = dataset_train[0]
    print(f"{type(sample)}")
    for i in sample:
        print(type(i))

    transform1 = transforms.ToTensor()
    dataset_train.transform = transform1
    dataset_test.transform = transform1

    print(type(dataset_test))
    sample = dataset_train[0]
    print(f"{type(sample)}")
    for i in sample:
        print(type(i))
    i,j = sample
    print(i.shape, i.dtype)
    '''
    # 1.1.3
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset_train.transform = transform2
    dataset_test.transform = transform2
    
    # 1.1.4
    
    train_data, val_data, train_labels, val_labels = train_test_split(dataset_test.data, dataset_test.targets, test_size=0.2, random_state=42)

    # 1.2
    class myNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(4,4), stride=1)
            self.avgpool = nn.AvgPool2d(kernel_size=(2,2),stride=2)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(4,4), stride=1)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=1)
            self.fc1 = nn.Linear(in_features=120, out_features=84)
            self.fc2 = nn.Linear(in_features=84, out_features=10)

        def forward(self,x):
            x = self.avgpool(nn.Tanh(self.conv1(x)))
            x = self.avgpool(nn.Tanh(self.conv2(x)))
            x = nn.Tanh(self.conv3(x))
            x = torch.flatten(x,1)
            x = nn.Tanh(self.fc1(x))
            x = nn.Softmax(self.fc2(x))
    
    # 1.3
    
    train_labels = F.one_hot(torch.Tensor(train_labels).long(), num_classes=10).to(torch.float32)
    dataset_train = TensorDataset(torch.Tensor(train_data),train_labels)

    val_labels = F.one_hot(torch.Tensor(val_labels).long()-1, num_classes=10).to(torch.float32)
    dataset_val = TensorDataset(torch.Tensor(val_data),val_labels)

    model = myNet()
    print(model)

    '''
    Code for bonus question
    '''
    for seed in range(10):
        torch.manual_seed(seed)
        # Train the models here
