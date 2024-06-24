'''
Assignment 2
Student: Fabian Gobet
'''
# *** Packges ***
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

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
    plt.savefig(set_name.replace(" ","_")+"_image_per_class")
    plt.show()

    fig, ax = plt.subplots(figsize=(10,5))
    fig.suptitle(set_name+" frequency per class")
    plt.bar(dataset.classes, counts)
    plt.xticks(fontsize=8)
    plt.ylabel("frequency")
    plt.savefig(set_name.replace(" ","_")+"_histogram")
    plt.show()

def plot_results(x,y1,y2,ylabel,label1,label2,title,save_name=None,show=False,color1='blue',color2='green'):
    fig, _ = plt.subplots(figsize=(7,5))
    fig.suptitle(title)
    plt.plot(x, y1, color=color1, label=label1)
    plt.plot(x, y2, color=color2, label=label2)
    plt.xlabel("steps")
    plt.ylabel(ylabel)
    plt.legend()
    if(save_name is not None):
        plt.savefig(save_name)
    if(show):
        plt.show()


def loss_acc_message(epoch,num_epochs,step,max_steps,train_loss,val_loss,train_acc,val_acc):
    print(f"Epoch {epoch}/{num_epochs}, Step {step}/{max_steps}")
    print(f"Train -> accuracy: {train_acc[-1]*100:.2f}, loss: {train_loss[-1]:.4f}")
    print(f"Validation -> accuracy: {val_acc[-1]*100:.2f}, loss: {val_loss[-1]:.4f}\n")


def checkpoint(model,optimizer,save_name=None):
    dic = {
        'model_state' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    if save_name is not None:
        torch.save(dic, save_name+".pt")
    return dic


def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, max_steps, n):
  
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    step_indices = [0]
    step = 0
    running = [0.0,0,0]

    with torch.no_grad():
        model.eval()
        for i, (d,t) in enumerate([next(iter(train_loader)), next(iter(val_loader))],0):
            d = d.to(DEVICE)
            t = t.to(DEVICE)
            t_yhat = model(d)
            _, pred = torch.max(t_yhat.data,1)
            corr = (pred == t).sum().item()
            t_loss = loss_fn(t_yhat,t)
            if i==0:
                train_loss.append(t_loss.item())
                train_acc.append(corr/len(t))
            else:
                val_loss.append(t_loss.item())
                val_acc.append(corr/len(t))

    loss_acc_message(0,num_epochs,step,max_steps,train_loss,val_loss,train_acc,val_acc)
    for epoch in range(1,num_epochs+1):
        for batch_num, (t_data,t_targets) in enumerate(train_loader,1):
            step = step + 1
            running[2] = running[2]+1
            model.train()
            t_data = t_data.to(DEVICE)
            t_targets = t_targets.to(DEVICE)
            t_yhat = model(t_data)
            t_loss = loss_fn(t_yhat,t_targets)
            running[0] = running[0] + t_loss.item()
            _, t_pred = torch.max(t_yhat.data,1)
            running[1] = running[1] + (t_pred == t_targets).sum().item()
            t_loss.backward()
            if step%n==0 or step==max_steps:
                step_indices.append(step)
                train_loss.append(running[0]/running[2])
                train_acc.append(running[1]/(running[2]*len(t_targets)))
                running = [0.0,0,0]
                with torch.no_grad():
                    model.eval()
                    for val_data, val_labels in val_loader:
                        val_data = val_data.to(DEVICE)
                        val_labels = val_labels.to(DEVICE)
                        v_yhat = model(val_data)
                        _, v_pred = torch.max(v_yhat.data,1)
                        v_correct = (v_pred == val_labels).sum().item()
                        val_acc.append(v_correct/len(val_labels))

                        v_loss = loss_fn(v_yhat,val_labels)
                        val_loss.append(v_loss.item())
                loss_acc_message(epoch,num_epochs,step,max_steps,train_loss,val_loss,train_acc,val_acc)
            optimizer.step()
            optimizer.zero_grad()
    return train_acc,val_acc,train_loss,val_loss,step_indices

def test_model(model,test_loader):
    with torch.no_grad():
        model.eval()
        for test_data, test_labels in test_loader:
            test_data = test_data.to(DEVICE)
            test_labels = test_labels.to(DEVICE)
            test_yhat = model(test_data)
            _, test_pred = torch.max(test_yhat,1)
            test_corr = (test_pred == test_labels).sum().item()
            acc = (test_corr*100)/len(test_labels)
            print(f"Test -> accuracy: {acc:.2f}\n")

if __name__ == "__main__":
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)


    # 1.1.1
    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    histo(dataset_train, "Cifar10 train set")
    histo(dataset_test, "Cifar10 test set")

# 1.1.2
    sample = dataset_train[0]
    print(f"\n1.1.2: dataset_train[0].shape -> type: {type(sample)}")
    for i in sample:
        print(f"1.1.2: type(dataset_train[0]) components -> type: {type(i)}")

    transform1 = transforms.ToTensor()
    dataset_train.transform = transform1
    dataset_test.transform = transform1


    sample = dataset_train[0]
    print(f"1.1.2: transform1(dataset_train[0]) -> type: {type(sample)}")
    for i in sample:
        print(f"1.1.2: type(transform1(dataset_train[0])) components -> type: {type(i)}")
    i,_ = sample
    print(f"1.1.2: type(transform1(dataset_train[0][0]))  -> shape: {i.shape}, dtype: {i.dtype}\n")

# 1.1.3
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform3 = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset_train.transform = transform3
    dataset_test.transform = transform2

# 1.1.4
    dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.2, random_state=42, shuffle=True)
    sample = dataset_train[0]
    print(f"1.1.2: transform1(dataset_train[0]) -> type: {type(sample)}")
    for i in sample:
        print(f"1.1.2: type(transform1(dataset_train[0])) components -> type: {type(i)}")
    print(dataset_train[0][0].shape)

# 1.2
    class myNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1,bias=False) # 32 16
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1,bias=False) # 16 8
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1,bias=False) # 8 4
            self.bn3 = nn.BatchNorm2d(64)

            self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

            self.fc1 = nn.Linear(4*4*64, 500, bias=False)
            self.fc2 = nn.Linear(500, 10, bias=False)

            self.drop1 = nn.Dropout(0.5)


        def forward(self,x):
            x = F.relu(self.conv1(x))
            x = self.bn1(x)
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.bn2(x)
            x = self.pool1(x)
            x = F.relu(self.conv3(x))
            x = self.bn3(x)
            x = F.relu(self.pool1(x))
            x = self.drop1(x)

            x = x.view(-1,4*4*64)
            x = F.relu(self.fc1(x))
            x = self.drop1(x)
            x = self.fc2(x)
            return x

# 1.3.1
    batch_size = 32
    num_epochs = 20
    learning_rate = 7e-4
    weight_decay = 1e-5
    max_steps = -((-len(dataset_train)*num_epochs)//batch_size)
    n = max_steps//25

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=len(dataset_val), num_workers=2)
    test_loader = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), num_workers=2)

    #cuda:0 because i was using colab, and the if statement is there because sometimes i ran out of resources
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myNet().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    print(f"1.3: {model}")
    print(f"1.3: {optimizer}")

    train_acc,val_acc,train_loss,val_loss,step_indices = train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, max_steps, n)

    test_model(model,test_loader)

    chkp = checkpoint(model,optimizer,"Fabian_Gobet_1")

# 1.3.4
    plot_results(step_indices,train_loss,val_loss,'loss','train loss','validation loss','Train/Validation losses',show=False, save_name="losses_1")
    plot_results(step_indices,torch.tensor(train_acc)*100,torch.tensor(val_acc)*100,'% accuracy','train accuracy','validation accuracy','Train/Validation accuracies',show=False,save_name="accs_1")

    '''
    Code for bonus question
    '''
    for seed in range(10):

        torch.manual_seed(seed)
        model = myNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_acc,val_acc,train_loss,val_loss,step_indices = train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, max_steps, n)

        plot_results(step_indices,train_loss,val_loss,'loss','train loss','validation loss','Train/Validation losses',show=False, save_name="losses_seed_"+str(seed))
        plot_results(step_indices,torch.tensor(train_acc)*100,torch.tensor(val_acc)*100,'% accuracy','train accuracy','validation accuracy','Train/Validation accuracies',show=False,save_name="accs_seed_"+str(seed))
        test_model(model,test_loader)
