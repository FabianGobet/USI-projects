'''
Assignment 2
Student: Fabian Gobet
'''
# *** Packges ***
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
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
    
    sample = dataset_train[0]
    print(f"\n1.1.2: dataset_train[0].shape -> type: {type(sample)}")
    for i in sample:
        print(f"1.1.2: type(dataset_train[0]) components -> type: {type(i)}")

    transform1 = transforms.ToTensor()
    dataset_train.transform = transform1
    dataset_test.transform = transform1

    print(f"1.1.2: transform1(dataset_train) -> type: {type(dataset_test)}")
    sample = dataset_train[0]
    print(f"1.1.2: transform1(dataset_train[0]) -> type: {type(sample)}")
    for i in sample:
        print(f"1.1.2: type(transform1(dataset_train[0])) components -> type: {type(i)}")
    i,_ = sample
    print(f"1.1.2: type(transform1(dataset_train[0][0]))  -> shape: {i.shape}, dtype: {i.dtype}\n")
    

    # 1.1.3
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset_train.transform = transform2
    dataset_test.transform = transform2
    

    # 1.1.4
    train_data, val_data, train_labels, val_labels = train_test_split(dataset_test.data, dataset_test.targets, test_size=0.2, random_state=42)

    print(f"1.1.4: train_data, train_labels -> type: {type(train_data)}, {type(train_labels)}")
    print(f"1.1.4: train_data -> shape: {train_data.shape}")

    train_data = torch.FloatTensor(train_data).permute(0,3,1,2)
    val_data = torch.FloatTensor(val_data).permute(0,3,1,2)
    train_labels = torch.IntTensor(train_labels)
    val_labels = torch.IntTensor(val_labels)

    print(f"1.1.4: train_data -> shape: {train_data.shape}, dtype: {train_data.dtype}")
    print(f"1.1.4: train_labels -> shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    print(f"1.1.4: val_data -> shape: {val_data.shape}, dtype: {val_data.dtype}")
    print(f"1.1.4: val_labels -> shape: {val_labels.shape}, dtype: {val_labels.dtype}\n")


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
            x = self.avgpool(nn.Tanh()(self.conv1(x)))
            x = self.avgpool(nn.Tanh()(self.conv2(x)))
            x = nn.Tanh()(self.conv3(x))
            x = torch.flatten(x,1)
            x = nn.Tanh()(self.fc1(x))
            return nn.Sigmoid()(self.fc2(x))

    

    # 1.3
    batch_size = 500 # 8000/batch_size = 8 batches
    num_epochs = 100 # num_epochs*num_batches = 800 no iterations
    learning_rate = 4e-3
    accum_iter = 1 # no_iterations/accum_iter = 160 weight updates
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    #train_labels = F.one_hot(train_labels, num_classes=10).to(torch.float32)
    dataset_train = TensorDataset(train_data,train_labels)

    #val_labels = F.one_hot(val_labels, num_classes=10).to(torch.float32)
    dataset_val = TensorDataset(val_data,val_labels)

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=len(dataset_val), shuffle=False)

    model = myNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    print(f"1.3: {model}")
    print(f"1.3: {optimizer}")

    train_accuracies = []
    val_accuracies = []
    train_loss = []
    val_loss = []
    n = 25
    iteration = 1


    for epoch in range(num_epochs):
        loss_val = 0.0
        correct_test = 0.0
        for i, data in enumerate(train_loader,1):
            model.train()
            inputs, targets = data
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            yhat = model(inputs)

            loss = loss_fn(yhat,targets)
            loss = loss/accum_iter
            loss_val = loss_val + loss.item()
            loss.backward()
            
            _, predicted = torch.max(yhat,1)
            correct_test = correct_test + (predicted == targets).sum().item()

            if(i%accum_iter==0 or i==len(train_loader)):
                train_loss.append(loss_val)
                optimizer.step()
                optimizer.zero_grad()
                train_accuracies.append(correct_test/(accum_iter*batch_size))
                loss_val = 0.0
                correct_test = 0.0
                
                with torch.no_grad():
                    model.eval()
                    for data, targets in val_loader: # only 1 batch
                        data = data.to(DEVICE)
                        targets = targets.to(DEVICE)

                        outputs = model(data)
                        loss = loss_fn(outputs,targets)
                        val_loss.append(loss.item())
                        _, predicted = torch.max(outputs,1)
                        correct_val = (predicted == targets).sum().item()
                        val_accuracies.append(correct_val/len(targets))

            if(iteration%n == 0):
                print(f"Epoch {epoch}, iteration {iteration}")
                print(f"Train -> accuracy: {train_accuracies[-1]*100:.2f}, loss: {train_loss[-1]:.4f}")
                print(f"Validation -> accuracy: {val_accuracies[-1]*100:.2f}, loss: {val_loss[-1]:.4f}\n")
            iteration = iteration + 1



    '''
    Code for bonus question
    '''
    for seed in range(10):
        torch.manual_seed(seed)
        # Train the models here
