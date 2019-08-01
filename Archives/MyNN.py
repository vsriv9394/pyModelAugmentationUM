import numpy as np
from KerasNN import Neural_Network as kNN
from plotting import myplot, myfigshow

class Neural_Network:
    '''

    Neural_Network <Class>: Class containing the Neural Network used to learn the features

    '''
    
    
    def __init__(self, NNconfig):
        '''

        __init__<Function>: Initialization for Neural_Network class

        - args:
          |- NNconfig <dictionary>: Contains the configuration for the Neural Network
             |- {"layers" : <int List>}: Contains the number of neurons in every hidden layer
             |- {"act_fn" : <str>}: Activation function for the neural network
             |- {"weights": <1-D NumPy array>}: Array containing all the weights and biases

        - kwargs:
          |- None

        - return value:
          |- None

        '''

        self.nNeurons         = np.asfortranarray(NNconfig["layers"])
        '''

        nNeurons <int List>: Number of neurons in all the hidden layers

        '''
        
        self.act_fn           = NNconfig["act_fn"]
        '''

        act_fn <str>: Activation function for the hidden layers

        '''

        self.weights          = np.asfortranarray(NNconfig["weights"])
        '''

        weights <1-D NumPy array>: Array containing all the weights and biases

        '''

        self.n_features       = NNconfig["n_features"]
        self.neurons          = np.asfortranarray(np.zeros((int(self.n_features+np.sum(self.nNeurons)+1))))
        self.features         = None


    
    def init_weights(self, features):
        
        self.features = np.asfortranarray(features)
        beta = np.zeros((np.shape(features)[0]))
        d_weights = np.asfortranarray(self.weights.copy())
        
        for i in range(10000):

            self.weights = self.weights - d_weights * 0.001 * 0.1**(0.25*np.log10(i+1.0))

            beta = MyNN.nn.predict(act_fn, self.nNeurons, self.weights, self.neurons, self.features)

            if i==0:
                myplot("ML_init", np.linspace(1,n_data,n_data), beta, '-b', 2.0, None)
            
            print("ML Iteration: %6d\t\tLoss: %E"%(i,eval_loss(beta)))

            d_beta    = 2.0*(beta-1.0)
            d_weights = MyNN.nn.get_sens(act_fn, self.nNeurons, self.weights, self.neurons, self.features, d_beta)

            d_weights = d_weights / np.max(np.abs(d_weights))
        
        myplot("ML_init", np.linspace(1,n_data,n_data), beta, '-g', 2.0, None)
        myfigshow("ML_init")



    def predict(self):
        
        return MyNN.nn.predict(act_fn, self.nNeurons, self.weights, self.neurons, self.features)



    def get_sens(self, d_beta):

        d_weights = MyNN.nn.get_sens(act_fn, self.nNeurons, self.weights, self.neurons, self.features, d_beta)
