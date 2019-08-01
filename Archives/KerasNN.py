import numpy as np
from plotting     import myfig, myplot, myscatter
from keras.models import Sequential
from keras.layers import Dense
from keras        import optimizers, backend

class Neural_Network:
    '''

    KerasNN <Class>: Class containing the keras neural network application

    '''
    
    
    def __init__(self, NNconfig):
        '''

        Initialization function <Function>

        - args:
          |- NNconfig <Dictionary>: Contains the neural network configuration
             |- {"layers" : <Integer List>}: Contains the number of neurons in each layer
             |- {"n_epochs" : <int>}: Number of training epochs
             |- {"batch_size" : <int>} : Number of inputs in each batch for stochastic training
             |- {"nfeatures" : <int>} : Number of features used to train the Neural Network
             |- {"validation_split" : <float between 0 and 1>} : Fraction of the data from the end of the training set
             |                                                   to be used only for validation
             |- {"lr" : <float>} : Learning rate for training

        - kwargs:
          |- None

        - return value:
          |- None

        '''
    
        self.nLayers          = np.shape(NNconfig["layers"])[0]
        '''
        
        nLayers <int>: Number of hidden layers in the neural network

        '''

        self.nNeurons         = NNconfig["layers"]
        '''
        
        nNeurons <List of int arrays>: Number of neurons in each hidden layer

        '''

        self.epochs_val       = NNconfig["n_epochs"]
        '''
        
        epochs_val <int> : Number of epochs for which to be trained

        '''

        self.batch_size_val   = NNconfig["batch_size"]
        '''

        batch_size_val <int>: Number of input data points to be used in one iteration during training

        '''

        self.nfeatures        = NNconfig["nfeatures"]
        '''

        nfeatures <int>: Number of features used for training

        '''

        self.validation_split = NNconfig["validation_split"]
        '''

        validation_split <float>: fraction of the data points from the end of training data for validation only

        '''

        self.optim            = optimizers.Adam(lr=NNconfig["lr"]) #(lr=0.2, decay=0.9999, amsgrad=False)
        '''

        optim <keras.optimizers>: Optimizer to be used during training

        '''
        
        self.model            = Sequential()
        '''

        model <keras.models>: Type of Neural Network Model to be used

        '''

        self.model.add( Dense( self.nNeurons[0], input_dim=self.nfeatures, kernel_initializer="uniform", activation="relu" ) ) 
        for iLayer in range(1,self.nLayers):
            self.model.add( Dense( self.nNeurons[iLayer], kernel_initializer="uniform", activation="relu" ) ) 
        self.model.add( Dense( 1, kernel_initializer="uniform", activation="linear" ) ) 
        self.model.compile(loss="mean_squared_error", optimizer=self.optim, metrics=["accuracy"])



    def getGradients(self, features):
        
        gradients = backend.gradients(self.model.output, self.model.trainable_weights)
        sess      = backend.get_session()
        evaluated_gradients = sess.run(gradients, feed_dict={self.model.input:features})
        return evaluated_gradients


    
    def train(self, beta, features, verbose=0, plot=False):
        '''

        train <Function>: Trains the Neural Network

        - args:
          |- beta <1-D NumPy array>: beta for training
          |- features <2-D NumPy array>: features for training
          |- verbose <int, 0 or 1>: Print the training statistics at every epoch
          |- plot <bool>: Whether to print the verification plot after training

        - kwargs:
          |- None

        - return value:
          |- None

        '''
    
        np.random.seed(8)
        self.model.fit(features, beta, epochs=self.epochs_val, batch_size=self.batch_size_val,\
                                       validation_split=self.validation_split, verbose=verbose)
        
        if plot==True:
    
            beta_new = self.model.predict(features, verbose=0)
            myplot(   "NN_train_verify", [np.min(beta), np.max(beta)], [np.min(beta), np.max(beta)], '-g', 2,      None)
            myscatter("NN_train_verify",                         beta,                     beta_new,  'b',         None)
            myfig(    "NN_train_verify",           "$$\\beta_{inv}$$",            "$$\\beta_{ML}$$", None, legend=False)

    
    
    def save_model(self, filename):
        '''

        save_model <Function>: Save the model weights to the given filename

        - args:
          |- filename <str>: Name of the file to which the weights have to be saved

        - kwargs:
          |- None

        - return value:
          |- None

        '''

        self.model.save_weights(filename)

    
    
    def load_model(self, filename):
        '''

        load_model <Function>: Load the model weights from the given filename

        - args:
          |- filename <str>: Name of the file from which the weights have to be loaded

        - kwargs:
          |- None

        - return value:
          |- None

        '''

        self.model.load_weights(filename)



    def set_weights_from_array(self, weights):
        
        start       = 0
        nrows       = self.nfeatures
        weight_list = []

        for iLayer in range(self.nLayers):
            
            ncols       = self.nNeurons[iLayer]
            weights_end = start + nrows * ncols
            biases_end  = weights_end + ncols
            weight_list.append(np.reshape(weights[start:weights_end],(nrows, ncols)))
            weight_list.append(weights[weights_end:biases_end])
            nrows   = ncols
            start   = biases_end

        ncols = 1
        weights_end = start + nrows
        biases_end  = weights_end + 1
        weight_list.append(np.reshape(weights[start:weights_end],(nrows, ncols)))
        weight_list.append(weights[weights_end:biases_end])

        self.model.set_weights(weight_list)



    def get_array_from_list(self, weight_list):
        
        for iLayer in range(self.nLayers+1):
            
            weight_list[2*iLayer  ] = np.reshape(weight_list[2*iLayer  ],(np.size(weight_list[2*iLayer  ])))
            weight_list[2*iLayer+1] = np.reshape(weight_list[2*iLayer+1],(np.size(weight_list[2*iLayer+1])))

        weights = np.hstack(weight_list)

        return weights
    
    

    def predict(self, features):
        '''

        predict <Function>: Predict the output of the model given features and weights

        - args:
          |- features <2-D NumPy array>: Features used for training the neural network model
          |- weights <1-D NumPy array>: Weights in the neural network

        - kwargs:
          |- None

        - return value:
          |- beta <1-D NumPy array>: Model augmentation parameters

        '''

        beta = np.zeros((np.shape(features)[0]))
        #self.set_weights_from_array(weights)
        beta = self.model.predict(features, verbose=0)
        return beta

if __name__=="__main__":

    NN = Neural_Network({'layers':[3,3,3], 'n_epochs':100, 'batch_size':50, 'nfeatures':3, 'validation_split':0.0, 'lr':0.01})
    features = np.random.random((100,3))
    NN.set_weights_from_array(NN.get_array_from_list(NN.model.get_weights()))
    evalgrad = NN.get_array_from_list(NN.getGradients(features))
    print(evalgrad)
