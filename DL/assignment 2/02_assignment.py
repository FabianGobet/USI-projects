'''
Assignment 2
Student: Fabian Gobet
'''
# *** Packges ***
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# *** Functions ***
def histo(dataset):
    labels = dataset.classes
    counts = [0 for _ in labels]
    dic = {c:None for c in labels}

    for img, lbl in dataset:
        if dic[labels[lbl]] is None:
            dic[labels[lbl]] = img
        counts[lbl] = counts[lbl]+1
        
    nplots = int(len(labels)/2)
    fig = plt.figure(figsize=(2, nplots))
    for i in range(1, nplots +1):
        fig.add_subplot(2, nplots, i)
        plt.imshow(dic[labels[i-1]])
    plt.show()
    plt.clf()

    plt.bar(dataset.classes, counts)
    plt.title("Histogram")
    plt.xlabel("Classes")
    plt.xticks(fontsize=6)
    plt.ylabel("No. images")
    plt.show()

if __name__ == "__main__":
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)

    # 1.1.1
    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    histo(dataset_train)
        

    '''
    Code for bonus question
    '''
    for seed in range(10):
        torch.manual_seed(seed)
        # Train the models here
