import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=5000, reg_rate= 0): #Default alpha= 0.01, default epochs= 5000
        self.alpha= learning_rate
        self.w= None #weights
        self.b= None #bias
        self.n= epochs #Number of epochs
        self.J_history= [] #Storing J history
        self.lamb = reg_rate 
        self.mean = None
        self.std = None
    def train(self, x, y):
        m, f = x.shape #number of elements, number of features
        y= y.reshape(-1,1)
        #Z-Score Normalization
        self.mean= np.mean(x, axis=0)
        self.std= np.std(x, axis=0) #Standard deviation
        self.std[self.std==0]= 1 #Trying to avoid 0
        xs= (x- self.mean)/self.std

        self.w = (np.random.randn(f,1)).reshape(-1,1)
        self.b = 0
        
        for i in range(self.n):
            fx= xs @ self.w+self.b
            tmpw= self.w-self.alpha*(xs.T @ (fx-y)+self.lamb*self.w)/m
            tmpb= self.b- self.alpha*(fx-y).sum()/m
            self.w= tmpw
            self.b= tmpb
            J= (np.sum((fx-y)**2)+self.lamb*np.sum((self.w)**2))/(2*m)
            self.J_history.append(J)
    def predict(self, x):
        xs = (x - self.mean) / self.std
        return xs @ self.w+self.b
class LogisticalRegression:
    def __init__(self, learning_rate=0.01, epochs=5000,reg_rate= 0): #Default alpha= 0.01, default epochs= 5000
        self.alpha= learning_rate
        self.w= None #weights
        self.b= None #bias
        self.n= epochs #Number of epochs
        self.J_history= [] #Storing J history
        self.lamb = reg_rate 
        self.mean = None
        self.std = None
    def train(self, x, y):
        m, f = x.shape #number of elements, number of features
        y= y.reshape(-1,1)
        #Z-Score Normalization
        self.mean= np.mean(x, axis=0)
        self.std= np.std(x, axis=0) #Standard deviation
        self.std[self.std==0]= 1 #Trying to avoid 0
        xs= (x- self.mean)/self.std

        self.w = (np.random.randn(f,1)).reshape(-1,1)
        self.b = 0
        for i in range(self.n):
            fx= xs @ self.w+self.b
            fy= 1/(1+np.exp(-fx))
            tmpw= self.w-self.alpha*(xs.T @ (fy-y)+self.lamb*self.w)/m
            tmpb= self.b- self.alpha*(fy-y).sum()/m
            self.w= tmpw
            self.b= tmpb
            L= -y*np.log(np.abs(fy)+1e-6)-(1-y)*np.log(np.abs(1-fy)+1e-6)
            J= L.sum()/m + self.lamb*np.sum((self.w)**2)/(2*m)
            self.J_history.append(J)
    def predict(self, x):
        xs= (x- self.mean)/self.std
        fx= xs @ self.w+self.b
        fy= 1/(1+np.exp(-fx))
        return((fy>0.5).astype(int))
    
class nn: #Neural Networks
    class DenseLayer:
        def __init__(self, n_inputs, n_neurons):
            # Initialize weights (random) and biases (zeros)
            self.w = np.random.randn(n_inputs, n_neurons) * 0.01
            self.b = np.zeros((1, n_neurons))
            self.inputs = None
            self.dw= None
            self.db=None
        def forward(self, inputs): #For PREDICTING
            self.inputs = inputs
            self.output = np.dot(inputs, self.w) + self.b #wx+b
        def backward(self, dvalues): #For LEARNING
            # Calculate Gradients
            # dW = inputs.T * dvalues
            self.dw = np.dot(self.inputs.T, dvalues)
            # db = sum(dvalues)
            self.db = np.sum(dvalues, axis=0, keepdims=True) #Keeps the dimensions same, keepdims
            # Gradient for the previous layer
            self.dinputs = np.dot(dvalues, self.w.T)
        def update_params(self, alpha):
            self.w -= alpha * self.dw
            self.b -= alpha * self.db
    class ReLU:
        def forward(self, inputs):
            self.inputs = inputs
            self.output = np.maximum(0, inputs) # 0 if less than 0, else same value

        def backward(self, dvalues):
            self.dinputs = dvalues.copy()
            # Derivative is 0 if input was <= 0
            self.dinputs[self.inputs <= 0] = 0
    class SoftmaxLoss:
        def forward(self, inputs, y_true):
            # 1. Softmax
            #Taking out the e^(wx+b) and dividing in numerator and denominator, so that it doesn't EXPLODE
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
            self.probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            
            return np.mean(-np.log(self.probs[range(len(y_true)), y_true] + 1e-7))

        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            # Shortcut: Gradient of Softmax + CrossEntropy is just (Probabilities - Truth)
            self.dinputs = self.probs.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples

class knn:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = np.array([
            self._euclidean_distance(x, x_train) #Distance to all training points
            for x_train in self.X_train
        ])

        k_indices = np.argsort(distances)[:self.k]  #Index values of k nearest points

        k_labels = self.y_train[k_indices]         # Labels of k nearest points
        values, counts = np.unique(k_labels, return_counts=True) #Majority DISTINCT points
        return values[np.argmax(counts)]
    
