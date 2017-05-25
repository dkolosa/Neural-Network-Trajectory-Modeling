import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    def __init__(self, X, y, parameters):
        #Input data
        self.X=X
        #Output data
        self.y=y
        #Expect parameters to be a tuple of the form:
        #    ((n_input,0,0), (n_hidden_layer_1, f_1, f_1'), ...,
        #     (n_hidden_layer_k, f_k, f_k'), (n_output, f_o, f_o'))
        self.n_layers = len(parameters)
        #Counts number of neurons without bias neurons in each layer.
        self.sizes = [layer[0] for layer in parameters]
        #Activation functions for each layer.
        self.fs =[layer[1] for layer in parameters]
        #Derivatives of activation functions for each layer.
        self.fprimes = [layer[2] for layer in parameters]
        self.build_network()

    def build_network(self):
        """ Builds the neural network by initializing all the wegiths"""
        #List of weight matrices taking the output of one layer to the input of the next.
        self.weights=[]
        #Bias vector for each layer.
        self.biases=[]
        #Input vector for each layer.
        self.inputs=[]
        #Output vector for each layer.
        self.outputs=[]
        #Vector of errors at each layer.
        self.errors=[]
        #We initialise the weights randomly, and fill the other vectors with 1s.
        for layer in range(self.n_layers-1):
            n = self.sizes[layer]
            m = self.sizes[layer+1]
            self.weights.append(np.random.normal(0, 1, (m, n)))
            # self.weights.append(np.random.normal(0.0, np.power(self.sizes,-0.5), (m, n)))
            self.biases.append(np.random.normal(0, 1, (m, 1)))
            self.inputs.append(np.zeros((n, 1)))
            self.outputs.append(np.zeros((n, 1)))
            self.errors.append(np.zeros((n, 1)))
        #There are only n-1 weight matrices, so we do the last case separately.
        n = self.sizes[-1]
        self.inputs.append(np.zeros((n, 1)))
        self.outputs.append(np.zeros((n, 1)))
        self.errors.append(np.zeros((n, 1)))

    def feedforward(self, x):
        """Propagates the input from the input layer to the output layer.
        """
        k=len(x)
        x.shape=(k,1)
        self.inputs[0]=x
        self.outputs[0]=x
        for i in range(1,self.n_layers):
            self.inputs[i]=self.weights[i-1].dot(self.outputs[i-1])+self.biases[i-1]
            self.outputs[i]=self.fs[i](self.inputs[i])
        return self.outputs[-1]

    def update_weights(self,x,y):
        """Update the weight matrices for each layer based on a single input x and target y.
        """
        output = self.feedforward(x)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(output-y)

        n=self.n_layers-2
        for i in range(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])
            self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]
        self.weights[0] = self.weights[0]-self.learning_rate*np.outer(self.errors[1],self.outputs[0])
        self.biases[0] = self.biases[0] - self.learning_rate*self.errors[1]

    def train(self, n_iter, learning_rate=1):
        """Train the network, Updates the weights after comparing each input in X with y
        repeat this process n_iter times."""

        self.learning_rate=learning_rate
        n=self.X.shape[0]
        for repeat in range(n_iter):
            #We shuffle the order in which we go through the inputs on each iter.
            index=list(range(n))
            np.random.shuffle(index)
            for row in index:
                x=self.X[row]
                y=self.y[row]
                self.update_weights(x,y)

    def predict_x(self, x):
        return self.feedforward(x)

    def predict(self, X):
        n = len(X)
        m = self.sizes[-1]
        ret = np.ones((n,m))
        for i in range(len(X)):
            ret[i,:] = self.feedforward(X[i])
        return ret


def logistic(x):
    return 1.0/(1+np.exp(-x))


def logistic_prime(x):
    ex = np.exp(-x)
    return ex/(1+ex)**2


def identity(x):
    return x


def identity_prime(x):
    return 1


def hyp_tan(x):
    return (2/(1+np.exp(-2*x))) - 1


def hyp_tan_prime(x):
    return 1 - (hyp_tan(x ** 2))
