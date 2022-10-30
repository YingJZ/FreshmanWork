import numpy as np
import pandas as pd
from layers import *

def MSEloss(x, y):
    return np.linalg.norm(x - y) ** 2

class Network :
    """
    hidden_layers 是一个由tuple组成的list，每个hidden layer是一个tuple，[0]表示类型，[1]表示神经元个数
    output_layer 同理
    """
    def __init__(self, input_layer_size, hidden_layers, output_layer, *, alpha = 0.001, adam = False, dropout = False, dropout_ratio = 0.3) : 
        self.inp_siz = input_layer_size
        self.layer_size = [x[1] for x in hidden_layers]
        self.hid_n = len(hidden_layers)
        self.adam = adam
        self.dropout = dropout
        self.alpha = alpha

        self.layers = []
        """
        for i in range(0, self.hid_n) :
            if i == 0 :
                self.layers.append(layer_kinds[hidden_layers[i][0]](input_layer_size, self.layer_size[i + 1]))
            elif i == self.hid_n - 1 :
                self.layers.append(layer_kinds[hidden_layers[i][0]](self.layer_size[i - 1], output_layer_size))
            else :
                self.layers.append(layer_kinds[hidden_layers[i][0]](self.layer_size[i - 1], self.layer_size[i + 1]))
        """
        for i in range(0, self.hid_n) :
            if i == 0 :
                self.layers.append(Affine(input_layer_size, self.layer_size[i], adam = self.adam, alpha = alpha))
                self.layers.append(layer_kinds[hidden_layers[i][0]]())
            else :
                self.layers.append(Affine(self.layer_size[i - 1], self.layer_size[i], adam = self.adam, alpha = alpha))
                self.layers.append(layer_kinds[hidden_layers[i][0]]())
            if self.dropout :
                self.layers.append(Dropout(dropout_ratio))
        
        self.layer_size.append(output_layer[1])
        self.layers.append(Affine(self.layer_size[self.hid_n - 1], output_layer[1], adam = self.adam, alpha = alpha))
        self.layers.append(layer_kinds[output_layer[0]]())
        
    def predict(self, X, *, training = True) :  # 前向传播
        for layer in self.layers :
            if type(layer) == Dropout :
               X = layer.forward(X, training)
            else :
               X = layer.forward(X)
        return X
    
    def gradient(self, X, y) : # 梯度下降
        y_hat = self.predict(X)
        layers = self.layers
        layers.reverse()
        last = True
        for layer in layers :
            if not last :
                grads = layer.backward(grads)
            else :
                grads = ((y_hat - np.array(y).reshape(y.shape[0],1)) / y.shape[0])
                last = False
        layers.reverse()
       
    def accuracy(self, X, y) :
        y_hat = np.array(self.predict(X, training = False)).flatten()  
        y = np.array(y)
        return (np.abs(y_hat - y) < 0.5).sum() / float(y.shape[0])


if __name__ == "__main__" :
#    X = np.array([[0,0],[0,1],[1,0],[1,1]])
#    testnet = Network(2, [("ReLU", 10), ("ReLU", 10)], ("Sigmoid", 1))
#    
#    for i in range(100) :
#        testnet.gradient(X, np.array([0, 1, 1, 0]))
#    
#    print("accuracy:", testnet.accuracy(X, np.array([0, 1, 1, 0])))

    df = pd.read_csv("processed_train.csv")

    df_train = df[:600]
    trainY = np.array(df_train["Survived"])
    trainX = np.array(df_train.drop(columns = ["Survived"]))

    df_test = df[600:]
    testY = np.array(df_test["Survived"])
    testX = np.array(df_test.drop(columns = ["Survived"]))

    feature_n = trainX.shape[1]
    testnet = Network(feature_n, [("ReLU", 100), ("ReLU", 100)], ("Sigmoid", 1), adam = True, dropout = True, dropout_ratio = 0.4)

    batch_size = 64
    for i in range(5000) :
        batch_mask = np.random.choice(trainX.shape[0], batch_size)
        batchX = trainX[batch_mask]
        batchY = trainY[batch_mask]
        testnet.gradient(batchX, batchY) 
        if i % 500 == 0 :
            print("train accuracy = ", testnet.accuracy(trainX, trainY), "test accuracy = ", testnet.accuracy(testX, testY))
        