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
"""
params = {"text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]} # Or palatino if you like it more
plt.rcParams.update(params)
"""
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

    is_p = True if color=="b" else False
    lab = "p(t)" if is_p else "predicted p(t)"
    lin_width = 1 if is_p else 1.5
    

    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.plot(z, y, color=color, label=lab, linewidth=lin_width)
    plt.xlabel('t')
    plt.ylabel('p(t)')
    #plt.legend()
    #plt.show()


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
    plt.legend()
    plt.show()
    
def create_batches(X,Y, batch_size=1, device=None, seed=42):
    
    number_training, _ = X.shape
    shuffle_state = np.random.RandomState(seed)
    indices = np.arange(number_training)
    shuffle_state.shuffle(indices)

    X_batches = []
    Y_batches = []

    for start_idx in range(0,number_training,batch_size):
        end_idx = min(start_idx + batch_size, number_training)
        batch_indices = indices[start_idx:end_idx]
        X_batch = torch.tensor(X[batch_indices], dtype=torch.float32)
        Y_batch = torch.tensor(Y[batch_indices], dtype=torch.float32)
        if device is not None:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)

    return X_batches, Y_batches

if __name__ == "__main__":


    # *** Question 3 **
    z_range = [-3,3]
    coeffs = np.array([0,-10,1,-1,1/100], dtype=np.float32).reshape(-1,1)
    coeffs_size = len(coeffs)
    x_train, y_train = create_dataset(coeffs, z_range, 500, 0.5, seed=0)
    x_eval, y_eval = create_dataset(coeffs, z_range, 500, 0.5, seed=1)
    #visualize_data(x_train,y_train,coeffs,z_range,title="training dataset")
    #visualize_data(x_eval,y_eval,coeffs,z_range,title="testing dataset")

    # *** Question 5 **
    DEVICE = torch.device("cuda:0" if torch.backends.mps.is_available() else "cpu")
    learning_rate = 0.5 #0.01 
    batch_size = 125
    num_epochs = 1000


    
    model = nn.Linear(coeffs_size,1,bias=False)
    loss_fn = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    x_eval = torch.tensor(x_eval, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32)

    model = model.to(DEVICE)
    x_eval = x_eval.to(DEVICE)
    y_eval = y_eval.to(DEVICE)

    batch_size = min(batch_size, x_train.shape[0])
    X_batches, Y_batches = create_batches(x_train,y_train, batch_size=batch_size, device=DEVICE, seed=42)

    train_loss = []
    eval_loss = []
    parameters = [model.weight.data.detach().cpu().numpy().squeeze().copy()]
    max_epochs = 1000
    num_iterations = 0

    print("-------------------------------------")
    print("Initial weights: {}".format(parameters[0]))
    print("Hyperparameters -> batch size: {}, learning rate: {}".format(batch_size,learning_rate))
    print("Optimizer: ",optimizer)
    print("Value in 1-vector x of size {}: {}".format(coeffs_size,model(torch.ones(coeffs_size))))
    print("-------------------------------------\n")

    start = time.time()
    for epoch in range(max_epochs):
        l = 1
        model.train()
        for x_batch, y_batch in zip(X_batches,Y_batches):
            num_iterations+=1
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = loss_fn(y_hat,y_batch)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()      
        

            model.eval()
            parameters.append(model.weight.data.detach().cpu().numpy().squeeze().copy())
            with torch.no_grad():
                y_hat_eval = model(x_eval)
                loss_eval = loss_fn(y_hat_eval,y_eval)
                l = loss_eval.item()
                eval_loss.append(l)
                if (num_iterations) % 200 == 0:
                    print("Number of iterations: ", num_iterations, "- Loss eval:", l)
                
                if l < 0.5:
                    break
        if l < 0.5:
            print("Training done after {} epochs, {} total iterations in time {} seconds".format(epoch+1,num_iterations, time.time() - start))
            print("Final evaluation loss: {}".format(l))
            break
        

    print("Final weights: {}\n".format(parameters[-1]))
    
    # *** Question 6 **
    """
    iteration_indices = np.linspace(1,num_iterations,num_iterations)
    plt.plot(iteration_indices,train_loss, color="blue", label="train")
    plt.plot(iteration_indices,eval_loss, color="red", label="evalutaion")
    plt.title("Train and evaluation losses over number of iterations")
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.yscale("log")
    plt.legend()
    plt.show()
    """
    # *** Question 7 **
    """
    plot_polynomial(parameters[-1], z_range, color='r')
    plot_polynomial(coeffs, z_range, color='b')
    plt.legend()
    plt.show()
    """ 
    # *** Question 8 **
    
    iterations_indices = np.arange(0,num_iterations+1)
    num_cols = coeffs_size//2 + coeffs_size%2
    fig, axs = plt.subplots(2, num_cols, squeeze=False)
    fig.suptitle("Weights variation over "+str(num_iterations)+" iterations.")
    for l in range(2):
        r = num_cols if (l+1)*num_cols <= coeffs_size else (coeffs_size%2)+1
        for c in range(num_cols):
            ax = axs[l,c]
            if(l*num_cols + c + 1 > coeffs_size):
                ax.set_visible(False)
            else:
                i = num_cols*l+c
                wi = np.array([vec[i] for vec in parameters], dtype=np.float32)
                ax = axs[l,c]
                ax.plot(iterations_indices, wi, color="blue")
                ax.set_title("w"+str(num_cols*l+c), fontsize=10)
    fig.tight_layout()
    plt.show()
    """      
    for i in range(coeffs_size):
        wi = [vec[i] for vec in parameters]
        plt.plot(iterations_indices, wi, label="w"+str(i), color=colors[i])
    plt.legend()
    plt.show()
    """
    # *** Question 9 **
    
    # *** Question 10 **