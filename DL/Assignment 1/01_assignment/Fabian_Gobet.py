'''
Assignment 1: Polynomial Regression
Student: NAME SURNAMES
'''
# Hint: to have nice figures in latex, where text has the same font as your document, use:
import matplotlib.pyplot as plt

"""
numpy has an optimized source code for matrix/vector operations, hence we should take advantage of it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

params = {"text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]} # Or palatino if you like it more
plt.rcParams.update(params)

"""
# A test (Uncomment to see the result)
plt.figure()
plt.xlabel(r'This is a CMS caption with some math: $\int_0^\infty f(x) dx$')
plt.show()
"""

# *** Question 1 **
def plot_polynomial(coeffs, z_range, color='b'):

    z_min, z_max = z_range
    sample_size = (z_max-z_min)*10
    z = np.linspace(z_min, z_max, num=sample_size).astype(np.float32)
    _,y = p(coeffs,z)

    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.plot(z,y,color=color, label="p(x)")
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.show()

    return 0

def p(coeffs,z):

    X = np.vander(z, N=len(coeffs), increasing=True).astype(np.float32)
    return X, (X@coeffs).astype(np.float32).reshape(-1,1)

# *** Question 2 **
def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):

    random_state = np.random.RandomState(seed)
    z_min, z_max = z_range
    z = random_state.uniform(z_min, z_max, (sample_size)).astype(np.float32)
    x,y = p(coeffs,z)
    y += random_state.normal(0.0, sigma, y.shape).astype(np.float32)
    return x,y

# *** Question 4 **
def visualize_data(X, y, coeffs, z_range, title=""):

    plt.scatter(X[:,1],y,color='red', marker="x", label = title, s=8)
    plt.title("p(x) and "+title)
    plot_polynomial(coeffs, z_range, color='b')
    

if __name__ == "__main__":


    # *** Question 3 **
    coeffs = np.array([0,-10,1,-1,1,1/100], dtype=np.float32).reshape(-1,1)
    z_range = [-3,3]
    x_train, y_train = create_dataset(coeffs, z_range, 500, 0.5, seed=0)
    x_test, y_test = create_dataset(coeffs, z_range, 500, 0.5, seed=1)
    #visualize_data(x_train,y_train,coeffs,z_range,title="training dataset")
    #visualize_data(x_test,y_test,coeffs,z_range,title="testing dataset")

    # *** Question 5 **
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = nn.Linear(len(coeffs),1,bias=False)
    loss_fn = nn.MSELoss()
    learning_rate = 0.8
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #print("Initial w:", model.weight)
    #print("Value in x = {1}:", model(torch.ones(len(coeffs))))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    #print(x_train.shape,"\n",x_test.shape,"\n",y_train.shape,"\n",y_test.shape)

    model.to(DEVICE)
    x_train = x_train.to(DEVICE)
    x_test = x_test.to(DEVICE)
    y_train = y_train.to(DEVICE)
    y_test = y_test.to(DEVICE)
    print(optimizer)

    """
    start = time.time()
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        y_hat = model(x_train)
        loss = loss_fn(y_hat,y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_hat_test = model(x_test)
            loss_test = loss_fn(y_hat_test,y_test)
            if (epoch+1) % 10 == 0:
                l = loss_test.item()
                print("Epoch:", epoch+1, "- Loss eval:", l)
                #print(model.weight,"\n")
                if l < 0.5:
                    print("Training done, with an evaluation loss of {} in time {}".format( loss_test.item(), time.time() - start))
                    break

    print("Final w:", model.weight)
    """
    # *** Question 6 **

    # *** Question 7 **

    # *** Question 8 **

    # *** Question 9 **

    # *** Question 10 **