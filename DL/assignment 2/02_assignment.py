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
    train_labels = torch.LongTensor(train_labels)
    val_labels = torch.LongTensor(val_labels)

    print(f"1.1.4: train_data -> shape: {train_data.shape}, dtype: {train_data.dtype}")
    print(f"1.1.4: train_labels -> shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    print(f"1.1.4: val_data -> shape: {val_data.shape}, dtype: {val_data.dtype}")
    print(f"1.1.4: val_labels -> shape: {val_labels.shape}, dtype: {val_labels.dtype}\n")


    # 1.2
    class myNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=18, kernel_size=(5,5), stride=1)
            self.conv2 = nn.Conv2d(in_channels=18, out_channels=30, kernel_size=(5,5), stride=1)
            self.pool1 = nn.AvgPool2d(kernel_size=(4,4),stride=1)

            self.conv3 = nn.Conv2d(in_channels=30, out_channels=45, kernel_size=(4,4), stride=1)
            self.conv4 = nn.Conv2d(in_channels=45, out_channels=60, kernel_size=(4,4), stride=1) 
            self.pool2 = nn.AvgPool2d(kernel_size=(4,4),stride=1) 
            self.conv5 = nn.Conv2d(in_channels=60, out_channels=75, kernel_size=(5,5), stride=1) 
            self.pool3 = nn.AvgPool2d(kernel_size=(5,5),stride=1)
            self.conv6 = nn.Conv2d(in_channels=75, out_channels=90, kernel_size=(4,4), stride=1) 
            self.fc1 = nn.Linear(in_features=90, out_features=50) 
            self.fc2 = nn.Linear(in_features=50, out_features=25) 
            self.fc3 = nn.Linear(in_features=25, out_features=10)
            
        def forward(self,x):
            tanh = nn.Tanh()
            x = self.pool1(F.tanh(self.conv2(F.tanh(self.conv1(x)))))
            x = self.pool2(F.tanh(self.conv4(F.tanh(self.conv3(x)))))
            x = self.pool3(F.tanh(self.conv5(x)))
            x = F.tanh(self.conv6(x))
            x = torch.flatten(x,1)
            x = F.tanh(self.fc1(x)) # change from sigmoid
            x = F.tanh(self.fc2(x))
            x = self.fc3(x)
            return x # change from Sigmoid

    
    
    # 1.3.1
    batch_size = 32 # 8000/64 = 125 batches -> total_batches = len(train_loader)/batch_size
    num_epochs = 30 # 125*10 = 1250 steps - > max_steps = num_epochs*len(train_loader)/batch_size
    learning_rate = 8e-5
    weight_decay = 1e-4
    max_steps = -((-len(train_data)*num_epochs)//batch_size)
    n = max_steps//50 
    
    dataset_train = TensorDataset(train_data,train_labels)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    val_data = val_data.to(DEVICE)
    val_labels = val_labels.to(DEVICE)
    model = myNet().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    #print(f"1.3: {model}")
    #print(f"1.3: {optimizer}")

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    step_indices = [0]
    step = 0
    

        # Initialize lists on step 0
    for i, (d,t) in enumerate([(train_data,train_labels), (val_data,val_labels)],0):
        with torch.no_grad():
            model.eval()
            d = d.to(DEVICE)
            t = t.to(DEVICE)
            t_yhat = model(d)
            t_loss = loss_fn(t_yhat,t)
            t_yhat = F.softmax(t_yhat)
            _, pred = torch.max(t_yhat,1)
            corr = (pred == t).sum().item()
            if i==0:
                train_loss.append(t_loss.item())
                train_acc.append(corr/len(t))
            else:
                val_loss.append(t_loss.item())
                val_acc.append(corr/len(t))
    
    # 1.3.2
    t_loss_val = 0.0
    t_correct = 0.0
    cycles = 0.0
    loss_acc_message(0,num_epochs,step,max_steps,train_loss,val_loss,train_acc,val_acc)
    for epoch in range(1,num_epochs+1):
        for batch_num, (t_data,t_targets) in enumerate(train_loader,1):
            step = step + 1
            cycles = cycles +1

            model.train()
            t_data = t_data.to(DEVICE)
            t_targets = t_targets.to(DEVICE)

            t_yhat = model(t_data)

            t_loss = loss_fn(t_yhat,t_targets)
            t_loss_val = t_loss_val + t_loss.item()
            t_loss.backward()
            
            _, t_pred = torch.max(t_yhat,1)
            t_correct = t_correct + (t_pred == t_targets).sum().item()

            optimizer.step()
            optimizer.zero_grad()

            if step%n==0 or step==max_steps:
                step_indices.append(step)
                train_loss.append(t_loss_val/cycles)
                train_acc.append(t_correct/(batch_size*cycles))
            
                with torch.no_grad():
                    model.eval()

                    v_yhat = model(val_data)
                    v_loss = loss_fn(v_yhat,val_labels)
                    val_loss.append(v_loss.item())

                    _, v_pred = torch.max(v_yhat,1)
                    v_correct = (v_pred == val_labels).sum().item()
                    val_acc.append(v_correct/len(val_labels))

                t_loss_val = 0.0
                t_correct = 0.0
                cycles = 0.0
                loss_acc_message(epoch,num_epochs,step,max_steps,train_loss,val_loss,train_acc,val_acc)
            
                    

            
            

    # 1.3.3
    checkpoint = {
        'model_state' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }

    #torch.save(checkpoint, 'Fabian_Gobet_1.pt')

    # 1.3.4 
    plot_results(step_indices,train_loss,val_loss,'loss','train loss','validation loss','Train/Validation losses',show=True)
    plot_results(step_indices,torch.tensor(train_acc)*100,torch.tensor(val_acc)*100,'% accuracy','train accuracy','validation accuracy','Train/Validation accuracies',show=True)

    '''
    Code for bonus question
    '''
    for seed in range(10):
        torch.manual_seed(seed)
        # Train the models here
