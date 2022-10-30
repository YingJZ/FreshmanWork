import numpy as np

class Affine :
    def __init__(self, pred, nxtd, *, adam = False, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, weight_decay = 1.00) :
        self.W = np.random.randn(pred, nxtd) * np.sqrt(2.0 / pred)
        self.b = np.zeros((1, nxtd))
        self.dW = np.zeros((pred, nxtd))
        self.db = np.zeros((1, nxtd))
        self.adam = adam
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        if self.adam :
            self.s = 0
            self.r = 0
            self.sb = 0
            self.rb = 0
            self.e = 0
        # print(pred, nxtd)

    def forward(self, X) :
        self._X = X
        return X @ self.W + self.b
    
    def backward(self, grads) :
        self.dW = self._X.T @ grads 
        self.db = np.array(grads.sum(axis = 0)).reshape(1, -1)
        self.update()
        return grads @ self.W.T
    
    def update(self) :
        if not self.adam :
            self.W = self.W * self.weight_decay - self.dW * self.alpha
            self.b = self.db * self.weight_decay - self.alpha
        else :
            self.e += 1
            beta1 = self.beta1
            beta2 = self.beta2
            e = self.e

            """
            self.s = (beta1 * self.s + (1 - beta1) * self.dW) / (1 - (beta1 ** e))
            self.r = (beta2 * self.r + (1 - beta2) * (self.dW ** 2)) / (1 - (beta2 ** e))
            self.W = self.W * self.weight_decay - self.alpha * self.s / (np.sqrt(self.r) + 1e-7)

            self.sb = (beta1 * self.sb + (1 - beta1) * self.db) / (1 - (beta1 ** e))
            self.rb = (beta2 * self.rb + (1 - beta2) * (self.db ** 2)) / (1 - (beta2 ** e))
            self.b = self.b * self.weight_decay - self.alpha * self.sb / (np.sqrt(self.rb) + 1e-7)
            """

            tmp = self.alpha * np.sqrt(1. - beta2 ** e) / (1. - beta1 ** e)
            self.s += (1 - beta1) * (self.dW - self.s)
            self.r += (1 - beta2) * (self.dW ** 2 - self.r)
            self.W -= tmp * self.s / (np.sqrt(self.r) + 1e-7)
            self.sb += (1 - beta1) * (self.db - self.sb)
            self.rb += (1 - beta2) * (self.db ** 2 - self.rb)
            self.b -= tmp * self.sb / (np.sqrt(self.rb) + 1e-7)


class ReLU :
    def __init__(self) :
        pass

    def forward(self, X) :
        self._X = X
        out = X.copy()
        out[X <= 0] = 0
        return out

    def backward(self, grads, alpha = 0.01) :
        new_grads = grads
        new_grads[self._X <= 0] = 0
        return new_grads

def sigmoid(X) :
    return np.exp(X) / (1 + np.exp(X))

class Sigmoid :
    def __init__(self) :
        pass

    def forward(self, X) :
        self._X = X
        return sigmoid(X)
    
    def backward(self, grads, alpha = 0.01) :
        t = sigmoid(self._X)
        return grads * t * (1 - t)

class Dropout :
    def __init__(self, ratio = 0.3) :
        self.ratio = ratio
    
    def forward(self, x, training = True) :
        if training :
            self.mask = np.random.rand(*x.shape) > self.ratio
            return x * self.mask
        else :
            return x * (1. - self.ratio)
    
    def backward(self, grads) :
        return grads * self.mask

layer_kinds = {"ReLU" : ReLU, "Sigmoid" : Sigmoid}
