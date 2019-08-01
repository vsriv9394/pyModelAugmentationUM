import MyNN
import numpy as np
from plotting import myplot, myfigshow

def eval_loss(beta):
    return sum((beta-1.0)**2)

def eval_loss_deriv(beta):
    return 2.0*(beta-1.0)

if __name__=="__main__":

    n_features = 3
    n_neurons_layer = [10,10,10]
    n_data = 200
    act_fn = "sigmoid"

    network = [n_features]
    network.extend(n_neurons_layer)
    network.append(1)
    network = np.array(network, order="F").astype(int)

    n_neurons_layer = np.array(n_neurons_layer, order="F").astype(int)

    n_weights = sum(network[1:]*(network[0:-1]+1))
    n_neurons = sum(network)

    neurons   = np.zeros((n_neurons), order="F")
    weights   = np.random.random((n_weights))
    d_weights = np.zeros((n_weights), order="F")

    features = np.random.random((n_features, n_data))

    for i in range(10000):

        weights = weights - d_weights * 0.001 * 0.1**(0.25*np.log10(i+1.0))

        #print(weights)

        #myplot(1, np.linspace(1,n_neurons,n_neurons), neurons, '-g', 2.0, None)
        beta = MyNN.nn.predict(act_fn, np.asfortranarray(n_neurons_layer),
                                       np.asfortranarray(weights),
                                       np.asfortranarray(neurons),
                                       np.asfortranarray(features))

        if i==0:
            myplot("ML_init", np.linspace(1,n_data,n_data), beta, '-b', 2.0, None)
        
        print("ML Iteration: %6d\t\tLoss: %E"%(i,eval_loss(beta)))

        #print("predicted")

        

        d_beta    = 2.0*(beta-1.0)
        d_weights = MyNN.nn.get_sens(act_fn, np.asfortranarray(n_neurons_layer),
                                             np.asfortranarray(weights),
                                             np.asfortranarray(neurons),
                                             np.asfortranarray(features),
                                             np.asfortranarray(d_beta))

        d_weights = d_weights / np.max(np.abs(d_weights))

        #print(d_weights)
    
    
    myplot("ML_init", np.linspace(1,n_data,n_data), beta, '-g', 2.0, None)
    myfigshow("ML_init")
