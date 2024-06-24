# -*- coding: utf-8 -*-
'''
Assignment 1: Polynomial Regression
Student: Fabian Flores Gobet
'''
# Hint: to have nice figures in latex, where text has the same font as your document, use:
import matplotlib.pyplot as plt
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
def p_func(coeffs,Z):
    X = np.vander(Z, N=len(coeffs), increasing=True).astype(np.float64)
    return X, (X@coeffs).astype(np.float64).reshape(-1,1)

def plot_polynomial(coeffs, z_range, color='b', func=p_func):
    z_min, z_max = z_range
    sample_size = 1000
    z = np.linspace(z_min, z_max, num=sample_size).astype(np.float64)
    _,y = func(coeffs,z)
    is_p = True if color=="b" else False
    lab = "f(x)" if is_p else "predicted f(x))"
    lin_width = 1 if is_p else 1.5
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.plot(z, y, color=color, label=lab, linewidth=lin_width)
    plt.xlabel('x')
    plt.ylabel('f(x)')


# *** Question 2 **
def create_dataset(coeffs, z_range, sample_size, sigma, seed=42, func=p_func):
    random_state = np.random.RandomState(seed)
    z_min, z_max = z_range
    Z = random_state.uniform(z_min, z_max, (sample_size)).astype(np.float64)
    x,y = func(coeffs,Z)
    y += random_state.normal(0.0, sigma, y.shape).astype(np.float64)
    return Z,x,y


# *** Question 4 **
def visualize_data(X, y, coeffs, z_range, title="", func=p_func):
    x = X[:,1] if X.ndim > 1 else X
    plt.scatter(x,y,color='red', marker="x", label = title, s=8)
    plt.title("f(x) and "+title)
    plot_polynomial(coeffs, z_range, color='b',func=func)
    plt.legend()
    plt.show()


# *** Helper functions **

def create_batches(X,Y, batch_size=1, device=None, seed=42):
    number_training = X.shape[0]
    shuffle_state = np.random.RandomState(seed)
    indices = np.arange(number_training)
    shuffle_state.shuffle(indices)

    X_batches = []
    Y_batches = []

    for start_idx in range(0,number_training,batch_size):
        end_idx = min(start_idx + batch_size, number_training)
        batch_indices = indices[start_idx:end_idx]
        X_batch = torch.tensor(X[batch_indices], dtype=torch.float64)
        Y_batch = torch.tensor(Y[batch_indices], dtype=torch.float64)
        if device is not None:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)

    return X_batches, Y_batches


def get_min_k_degree(a,error_base_10=5e-1):
  error = np.log10(error_base_10)
  k = 1
  R = float(5*(a**(k+1))/(k+1))

  while(np.log10(R)>=error):
    k += 1
    R = (R*a)/(k+1)
  return k


def sine_func(dummy_coeffs,Z):
  num_coffs = 2*len(dummy_sin_coeffs)-1
  vand_even_cols = np.vander(Z, N=num_coffs, increasing=True).astype(np.float64)[:,1::2]
  vand_even_cols = np.hstack((np.full((len(Z), 1), 3, dtype=np.float64), vand_even_cols))

  return vand_even_cols, (5*np.sin(Z)+3).astype(np.float64).reshape(-1,1)


if __name__ == "__main__":
    
    # *** Question 3 **
    z_range = [-3,3]
    coeffs = np.array([0,-10,1,-1,1/100], dtype=np.float64).reshape(-1,1)
    coeffs_size = len(coeffs)

    _, x_train, y_train = create_dataset(coeffs, z_range, 500, 0.5, seed=0)
    _, x_eval, y_eval = create_dataset(coeffs, z_range, 500, 0.5, seed=1)

    visualize_data(x_train, y_train, coeffs, z_range, title="training dataset")
    visualize_data(x_eval, y_eval, coeffs, z_range, title="testing dataset")


# *** Question 5 **
    DEVICE = torch.device("cuda:0" if torch.backends.mps.is_available() else "cpu")
    learning_rate = 0.5
    batch_size = 250
    num_epochs = 1000
    iterations_limit = 1000

    model = nn.Linear(coeffs_size,1,bias=False, dtype=torch.float64)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x_eval = torch.tensor(x_eval, dtype=torch.float64)
    y_eval = torch.tensor(y_eval, dtype=torch.float64)

    model = model.to(DEVICE)
    x_eval = x_eval.to(DEVICE)
    y_eval = y_eval.to(DEVICE)

    batch_size = min(batch_size, x_train.shape[0])
    X_batches, Y_batches = create_batches(x_train,y_train, batch_size=batch_size, device=DEVICE, seed=42)

    train_loss = []
    eval_loss = [loss_fn(model(x_eval),y_eval).item()]
    parameters = [model.weight.data.detach().cpu().numpy().squeeze().copy()]
    num_iterations = 0

    print("-------------------------------------")
    print("Initial weights: {}".format(parameters[0]))
    print("Hyperparameters -> batch size: {}, learning rate: {}".format(batch_size,learning_rate))
    print("Optimizer: ",optimizer)
    print("Value in 1-vector x of size {}: {}".format(coeffs_size,model(torch.ones(coeffs_size,dtype=torch.float64))))
    print("Initial evaluation loss: ",eval_loss[0])
    print("-------------------------------------\n")

    start = time.time()
    for epoch in range(num_epochs):
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

                if l < 0.5 or num_iterations >= iterations_limit:
                    break
        if l < 0.5 or num_iterations >= iterations_limit:
          print("\n-------------------------------------")
          print("Training done after {} epochs, {} total iterations in time {} ms".format(epoch+1,num_iterations, time.time() - start))
          print("Final evaluation loss: {}".format(l))
          break


    print("Final weights: {}".format(parameters[-1]))
    print("-------------------------------------\n")


# *** Question 6 **
    plt.plot(range(1,num_iterations+1),train_loss, color="blue", label="train")
    plt.plot(range(0,num_iterations+1),eval_loss, color="red", label="evalutaion")
    plt.title("Train and evaluation losses over number of iterations")
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.yscale("log")
    plt.legend()
    plt.show()


# *** Question 7 **
    plot_polynomial(parameters[-1], z_range, color='r')
    plot_polynomial(coeffs, z_range, color='b')
    plt.legend()
    plt.show()


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
                wi = np.array([vec[i] for vec in parameters], dtype=np.float64)
                ax = axs[l,c]
                ax.plot(iterations_indices, wi, color="blue")
                ax.set_title("w"+str(num_cols*l+c), fontsize=10)
    fig.tight_layout()
    plt.show()


# *** Question 9 **
    _, x_train_10, y_train_10 = create_dataset(coeffs, z_range, 10, 0.5, seed=0)
    _, x_eval_10, y_eval_10 = create_dataset(coeffs, z_range, 500, 0.5, seed=1)
    #visualize_data(x_train_10,y_train_10,coeffs,z_range,title="training dataset")
    #visualize_data(x_eval_10,y_eval_10,coeffs,z_range,title="testing dataset")

    model_10 = nn.Linear(coeffs_size,1,bias=False, dtype=torch.float64)
    loss_fn_10 = nn.MSELoss()

    DEVICE = torch.device("cuda:0" if torch.backends.mps.is_available() else "cpu")
    optimizer_10 = optim.Adam(model_10.parameters(), lr=learning_rate)
    x_eval_10 = torch.tensor(x_eval_10, dtype=torch.float64)
    y_eval_10 = torch.tensor(y_eval_10, dtype=torch.float64)

    model_10 = model_10.to(DEVICE)
    x_eval_10 = x_eval_10.to(DEVICE)
    y_eval_10 = y_eval_10.to(DEVICE)

    batch_size_10 = min(batch_size, x_train_10.shape[0])
    X_batches_10, Y_batches_10 = create_batches(x_train_10,y_train_10, batch_size=batch_size_10, device=DEVICE, seed=42)

    train_loss_10 = []
    eval_loss_10 = []
    parameters_10 = [model_10.weight.data.detach().cpu().numpy().squeeze().copy()]
    num_iterations_10 = 0

    print("-------------------------------------")
    print("Initial weights: {}".format(parameters_10[0]))
    print("Hyperparameters -> batch size: {}, learning rate: {}".format(batch_size_10,learning_rate))
    print("Optimizer: ",optimizer_10)
    print("Value in 1-vector x of size {}: {}".format(coeffs_size,model_10(torch.ones(coeffs_size,dtype=torch.float64))))
    print("-------------------------------------\n")

    start = time.time()
    for epoch in range(num_epochs):
        l = 1
        model_10.train()
        for x_batch, y_batch in zip(X_batches_10,Y_batches_10):
            num_iterations_10+=1
            optimizer_10.zero_grad()
            y_hat = model_10(x_batch)
            loss = loss_fn_10(y_hat,y_batch)
            train_loss_10.append(loss.item())
            loss.backward()
            optimizer_10.step()


            model_10.eval()
            parameters_10.append(model_10.weight.data.detach().cpu().numpy().squeeze().copy())
            with torch.no_grad():
                y_hat_eval = model_10(x_eval_10)
                loss_eval = loss_fn_10(y_hat_eval,y_eval_10)
                l = loss_eval.item()
                eval_loss_10.append(l)
                if (num_iterations_10) % 200 == 0:
                    print("Number of iterations: ", num_iterations_10, "- Loss eval:", l)

                if l < 0.5 or num_iterations >= iterations_limit:
                    break
        if l < 0.5 or num_iterations >= iterations_limit:
          print("\n-------------------------------------")
          print("Training done after {} epochs, {} total iterations in time {} seconds".format(epoch+1,num_iterations_10, time.time() - start))
          print("Final evaluation loss: {}".format(l))
          break


    print("Final weights: {}".format(parameters_10[-1]))
    print("-------------------------------------\n")

    iteration_indices_10 = np.linspace(1,num_iterations_10,num_iterations_10)
    plt.plot(iteration_indices_10,train_loss_10, color="blue", label="train")
    plt.plot(iteration_indices_10,eval_loss_10, color="red", label="evalutaion")
    plt.title("Train and evaluation losses over number of iterations")
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.yscale("log")
    plt.legend()
    plt.show()

    plot_polynomial(parameters_10[-1], z_range, color='r')
    plot_polynomial(coeffs, z_range, color='b')
    plt.legend()
    plt.show()

    
# *** Question 10 **
    a = 5
    a_z_range = [-a, a]
    max_error = 5e-1
    degree = get_min_k_degree(a,max_error)
    degree = degree if degree%2 != 0 else degree+1
    num_coeffs_sin = int((degree+1)/2 + 1)
    dummy_sin_coeffs = np.empty((num_coeffs_sin,1))

    Z_train_sin, x_train_sin, y_train_sin = create_dataset(dummy_sin_coeffs, a_z_range, 1000, 0.5, seed=0, func = sine_func)
    Z_eval_sin, x_eval_sin, y_eval_sin = create_dataset(dummy_sin_coeffs, a_z_range, 500, 0.5, seed=1,  func = sine_func)

    visualize_data(Z_train_sin, y_train_sin, None, a_z_range, title="training dataset", func=sine_func)
    visualize_data(Z_eval_sin, y_eval_sin, None, a_z_range, title="evaluation dataset", func=sine_func)

    DEVICE = torch.device("cuda:0" if torch.backends.mps.is_available() else "cpu")
    learning_rate_sin = 2.3e-5
    w_decay = 4e-4 
    batch_size_sin = 500
    num_epochs_sin = 32500

    model_sin = nn.Linear(num_coeffs_sin,1,bias=False, dtype=torch.float64)
    loss_fn_sin = nn.L1Loss(reduction='mean')
    optimizer_sin = optim.Adam(model_sin.parameters(), lr=learning_rate_sin, weight_decay=w_decay)

    x_eval_sin = torch.tensor(x_eval_sin, dtype=torch.float64)
    y_eval_sin = torch.tensor(y_eval_sin, dtype=torch.float64)

    model_sin = model_sin.to(DEVICE)
    x_eval_sin = x_eval_sin.to(DEVICE)
    y_eval_sin = y_eval_sin.to(DEVICE)

    batch_size_sin = min(batch_size_sin, x_train_sin.shape[0])
    X_batches_sin, Y_batches_sin = create_batches(x_train_sin, y_train_sin, batch_size=batch_size_sin, device=DEVICE, seed=42)

    train_loss_sin = []
    eval_loss_sin = []
    parameters_sin = [model_sin.weight.data.detach().cpu().numpy().squeeze().copy()]
    num_iterations_sin = 0

    print("-------------------------------------")
    print("Initial weights: {}".format(parameters_sin[0]))
    print("Hyperparameters -> batch size: {}".format(batch_size_sin))
    print("Optimizer: ",optimizer_sin)
    print("Value in 1-vector x of size {}: {}".format(num_coeffs_sin,model_sin(torch.ones(num_coeffs_sin,dtype=torch.float64))))
    print("-------------------------------------\n")

    start = time.time()
    for epoch in range(num_epochs_sin):
        l = 1e20
        model_sin.train()
        for x_batch, y_batch in zip(X_batches_sin,Y_batches_sin):
            num_iterations_sin += 1
            optimizer_sin.zero_grad()
            y_hat = model_sin(x_batch)
            loss = loss_fn_sin(y_hat,y_batch)
            if (num_iterations_sin % 1000) == 0:
                  train_loss_sin.append(loss.item())
            loss.backward()
            optimizer_sin.step()

            model_sin.eval()
            #parameters_sin.append(model_sin.weight.data.detach().cpu().numpy().squeeze().copy())

            with torch.no_grad():
                y_hat_eval = model_sin(x_eval_sin)
                loss_eval = loss_fn_sin(y_hat_eval,y_eval_sin)
                l = loss_eval.item()
                if (num_iterations_sin % 1000) == 0:
                  eval_loss_sin.append(l)
                if (num_iterations_sin % 5000) == 0:
                    print("Epoch: {}, Iteration: {}, Loss train: {:.2f}, - Loss eval: {:.2f}".format(epoch+1, num_iterations_sin, loss.item(), l))


                if l < max_error: # or num_iterations_sin >= iterations_limit_sin:
                    break
        if l < max_error: # or num_iterations_sin >= iterations_limit_sin:
          print("\n-------------------------------------")
          print("Training done after {} epochs, {} total iterations in time {} seconds".format(epoch+1,num_iterations_sin, time.time() - start))
          print("Final evaluation loss: {}".format(l))
          break


    print("Final weights: {}".format(model_sin.weight.data.detach().cpu().numpy().squeeze()))
    print("-------------------------------------\n")

iteration_indices_sin = np.linspace(1,num_iterations_sin,num_iterations_sin//1000)
plt.plot(iteration_indices_sin,train_loss_sin, color="blue", label="train")
plt.plot(iteration_indices_sin,eval_loss_sin, color="red", label="evalutaion")
plt.title("Train and evaluation losses along iterations for sin(x)")
plt.xlabel('iterations')
plt.ylabel('loss')
plt.yscale("log")
plt.legend()
plt.show()

x_sin_plot = np.linspace(-a,a,num=1000)
_ , y_sin_plot = sine_func(np.empty((1,1)), x_sin_plot)
plt.plot(x_sin_plot,y_sin_plot, color='b', label = "sin(x)", linewidth = 4)
plot_polynomial(parameters_sin[-1], a_z_range, color='r', func=sine_func)
plt.title("sin(x) and predicted McLaurin polynomial for sin(x)")
plt.legend()
plt.show()