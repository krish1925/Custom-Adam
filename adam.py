import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x,w,b):
    sigmoidd = 1/(1+np.exp(-(w*x+b)))
    return sigmoidd
def mse(x,y,w,b):
    L = 0.0
    for i in range(x.shape[0]):
        L += 0.5*(y[i]-sigmoid(x[i],w,b))**2
    return L
def grad_w_mse(x,y,w,b):
    fx = sigmoid(x,w,b) 
    deltaw = (fx - y)*fx*(1-fx)*x
    return deltaw
def grad_b_mse(x,y,w,b):
    fx = sigmoid(x,w,b) 
    deltab = (fx - y)*fx*(1-fx)
    return deltab


def cross_entropy(x,y,w,b):
    L = 0.0
    for i in range(x.shape[0]):
        L += -(y[i]*np.log(f(x[i],w,b)))
    return L
def grad_w_cross(x,y,w,b):
    fx = sigmoid(x,w,b) 
    deltaw = (- y)*(1-fx)*x
    return deltaw
def grad_b_cross(x,y,w,b):
    fx = sigmoid(x,w,b) 
    db = (- y)*(1-fx)
    return deltab


def select_loss_gradients(losstype):
    if(losstype == "CE"):
        loss = cross_entropy
        grad_w = grad_w_cross
        grad_b = grad_b_cross
    else: #(losstype == "MSE"):
        loss = mse
        grad_w = grad_w_mse
        grad_b = grad_b_mse   
    return loss, grad_w, grad_b


def regularization(w, b, lambdaa): #using l2 here jdskfaldsjnflk
    lambda_reg = lambdaa
    w -= lambdaa * w
    b -= lambdaa * b
    return w, b

def step_decay_lr_schedule(epoch, initial_lr=0.1, drop_factor=0.5, epochs_drop=10):
    return initial_lr * math.pow(drop_factor, math.floor((1 + epoch) / epochs_drop))

def Adam(x, y, epochs, batch_size, loss, lr,lambdaa, clip_value=None, lr_schedule=None ):
    w = 10 #np.random.randn()  #assign random weight and bias terms(may need to change if x is a vector of features)
    b = 10 #np.random.randn()
    epsilon = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    momentum_w, momentum_b = 0, 0
    update_w, update_b = 0, 0
    l_list = []
    w_list = []
    b_list = []
    m_correction = 1
    v_correction = 1
    

    loss_func, grad_w_func, grad_b_func = select_loss_gradients(loss)
    
    for i in range(1, epochs + 1):
        for j in range(0, x.shape[0], batch_size):
            x_batch = x[j:j+batch_size]
            y_batch = y[j:j+batch_size]
            dw = grad_w_func(x_batch, y_batch, w, b)
            db = grad_b_func(x_batch, y_batch, w, b)
            if clip_value is not None:
                dw = np.clip(dw, -clip_value, clip_value)
                db = np.clip(db, -clip_value, clip_value)
            
            momentum_w = beta1 * momentum_w + (1 - beta1) * dw #momentum 
            momentum_b = beta1 * momentum_b + (1 - beta1) * db
            

            update_w = beta2*update_w + (1-beta2) * dw**2   #update history
            update_b = beta2*update_b + (1-beta2) * db**2
            m_correction *= beta1
            v_correction *= beta2
            momentum_w_corr = momentum_w / (1 - m_correction)
            momentum_b_corr = momentum_b / (1 - m_correction)
            update_w_corr = update_w / (1 - v_correction)
            update_b_corr = update_b / (1 - v_correction)
            #lr scheduling
            if lr_schedule is not None:
                lr = lr_schedule(i, initial_lr=lr)
            #param update
            w -= (lr / (np.sqrt(update_w_corr) + epsilon)) * momentum_w_corr
            b -= (lr / (np.sqrt(update_b_corr) + epsilon)) * momentum_b_corr
        
        w, b = regularization(w, b, lambdaa)
        
        current_loss = loss_func(x, y, w, b)[0]
        print(f'Loss after {i}th epoch = {current_loss}\n')
        l_list.append(current_loss)
        w_list.append(w)
        b_list.append(b)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch Curve - Mini Batch Adam\nBatch Size = {batch_size}\n Starting learning rate = {lr}\nLoss Function = {loss}')
    plt.plot(range(1, epochs + 1), l_list)
    plt.show()
    
    return w_list, b_list, l_list


def test():
    x = np.linspace(-1, 1, 100) # Generating random points for testing
    true_w, true_b = 2, -1
    y = sigmoid(x, true_w, true_b) 
    #initialize here
    epochs = 100
    batch_size = 10
    loss_type = "MSE" #or CE for multiclass
    learning_rate = 0.1
    lambda_reg = 0.01  #regularization coefficient # change if needed
    loss_func, grad_w_func, grad_b_func = select_loss_gradients(loss_type)
    w_list, b_list, l_list = Adam(x, y, epochs, batch_size, loss_func, learning_rate, lambda_reg)
    plt.legend()
    plt.show()
test()
