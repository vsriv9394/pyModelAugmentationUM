import nn
import time
import numpy as np
from plotting import myplot, myfigshow

def eval_loss(features, beta):
    #return sum((beta-2.0)**2)
    return sum((beta-features[0,:]*features[1,:])**2)

def eval_loss_deriv(features, beta):
    #return 2.0*(beta-2.0)
    return 2.0*(beta-features[0,:]*features[1,:])

def init_weights(n_features, n_neurons, kind):
    network = [n_features]
    network.extend(n_neurons)
    network.append(1)
    network = np.array(network, order="F").astype(int)

    n_weights = sum(network[1:]*(network[0:-1]+1))

    if kind=="random":
        return np.asfortranarray(np.random.random((n_weights))), np.asfortranarray(network)
    else:
        return np.asfortranarray(np.loadtxt(kind)), np.asfortranarray(network)

def OptimizationUpdate(weights, d_weights, m_adam, v_adam, alpha, i_epoch):
    m_adam  = 0.9*m_adam + 0.1*d_weights
    v_adam  = 0.999*v_adam + 0.001*d_weights**2
    m_temp  = m_adam/(1.0-0.9**(i_epoch+1))
    v_temp  = v_adam/(1.0-0.999**(i_epoch+1))
    weights = weights - alpha*m_temp/(v_temp**0.5+1e-8)
    return weights, m_adam, v_adam

if __name__=="__main__":

    n_neurons = [5,5,5,5]
    n_data = 200
    act_fn = "sigmoid"

    features = np.zeros((1,n_data))
    features[0,:] = np.linspace(0., 2.0*np.pi, n_data)
    #features[1,:] = np.linspace(2, n_data+1, n_data)
    features = np.asfortranarray(features)
    #features = np.asfortranarray(np.random.random((3, n_data)))

    n_features = np.shape(features)[0]

    beta_target = np.sin(features[0,:])

    weights, network = init_weights(n_features, n_neurons, "random")
    d_weights  = np.zeros_like(weights)

    act_fn_name  = "sigmoid"
    loss_fn_name = "mse"
    opt_name     = "adam"
    batch_size   = 200
    n_epochs     = 30000
    train_fraction = 1.0
    opt_params   = np.asfortranarray(np.array([0.001, 0.9, 0.999, 1e-8, 1.0, 1.0]))
    
    fortran_train = True

    train_end = int(n_data*train_fraction)
    alpha     = 1e-3

    m_adam    = 0.0
    v_adam    = 0.0
    d_weights = 0.0

    #weights = weights*1e-5

    t1 = time.time()

    if fortran_train==True:

        nn.nn.nn_train(network,act_fn_name,loss_fn_name,opt_name,weights,features,beta_target,batch_size,n_epochs,train_fraction,opt_params)
        print(opt_params)

    else:

        for i_epoch in range(n_epochs):

            batch_start = 1

            while batch_start<=train_end:

                batch_end = (batch_start - 1) + batch_size

                if batch_end > train_end:
                    batch_end = train_end + 0

                #weights = weights - d_weights * alpha * 0.1**(0.35*np.log10(i_epoch+1.0))

                weights, m_adam, v_adam = OptimizationUpdate(weights,d_weights,m_adam,v_adam,alpha,i_epoch)

                beta = nn.nn.nn_predict(network, act_fn_name, loss_fn_name, opt_name, weights, features, opt_params)

                if i_epoch==0 and batch_start==1:
                    myplot("ML_init", np.linspace(1,n_data,n_data), beta, '-ob', 2.0, None)
                
                d_beta    = eval_loss_deriv(features, beta)
                d_weights = nn.nn.nn_get_weights_sens(network, act_fn_name, loss_fn_name, opt_name, weights, features, batch_start, batch_end, d_beta, opt_params)

                #print(np.linalg.norm(d_weights))

                batch_start = batch_end + 1

                #d_weights = d_weights / np.max(np.abs(d_weights))

            print("ML Iteration: %6d\t\tLoss: %E"%(i_epoch,eval_loss(features, beta)))

    t2 = time.time()

    print("Time = %E"%(t2-t1))

    beta = nn.nn.nn_predict(network, act_fn_name, loss_fn_name, opt_name, weights, features, opt_params)
    myplot("ML_init", np.linspace(1,n_data,n_data), beta, '-og', 2.0, None)
    myplot("ML_init", np.linspace(1,n_data,n_data), beta_target, '-or', 2.0, None)
    myfigshow()
