import numpy as np
from tensor import Tensor as T, dot

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            # or: p.grad = np.zeros_like(p.data)
    
    def step(self, lr):
        # sgd update step
        for p in self.parameters():
            p.data = p.data - lr * p.grad
    
    def parameters(self):
        return []
    
class LinearLayer(Module):
    def __init__(self, nin, nout, bias=True, nonlin=None) -> None:
        super().__init__()
        k = np.sqrt(1/nin)
        self.w = T.uniform(-k,k,(nout,nin))
        if bias:
            self.b = T.uniform(-k,k,(1,nout))
        self.bias = bias
        self.nonlin = nonlin
    
    def __call__(self, x):
        act = dot(x, T.transpose(self.w, (-1,-2)))
        if self.bias:
            act = act + self.b
        return self.nonlin(act) if self.nonlin else act
    
    def parameters(self):
        if self.bias:
            return [self.w, self.b]
        return [self.w]
    

class BatchNorm1D(Module):
    # input of shape (N,D)
    #https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
    def __init__(self, num_features, momentum=0.1) -> None:
        self.num_features = num_features
        self.gamma = T.ones(num_features)
        self.beta = T.zeros(num_features)
        self.momentum = momentum
        
        self.running_mean = T.zeros(num_features)
        self.running_var = T.ones(num_features)
        
    def __call__(self, x, training=True):
        # x is of shape (N, num_features)
        # or maybe not dont know
        # mean and var along axis=0
        if training:
            m = T.mean(x, axis=0, keepdims=True)
            v = T.var(x, axis=0, keepdims=True) + 1e-5
            
            # running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * m
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * v
            
            return self.gamma*((x - m)/T.sqrt(v)) + self.beta
        # testing
        return self.gamma/self.running_var * x + (self.beta - (self.gamma*self.running_mean)/self.running_var)
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    
class LayerNorm(Module):
    # input of shape (N,D)
    # or other i think
    def __init__(self, normalized_shape):
        # normalized_shape is equivalent to num_features for input in form (N,num_features)
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, tuple):
            self.normalized_shape = normalized_shape
        
        # i think this is correct but might not be
        self.axis_tuple = tuple([i for i in range(1, len(self.normalized_shape)+1)])
        
        self.gamma = T.ones(normalized_shape)
        self.beta = T.zeros(normalized_shape)
        
    def __call__(self, x):
        # x is of shape normalized_shape
        m = T.mean(x, axis=self.axis_tuple, keepdims=True)
        v = T.var(x, axis=self.axis_tuple, keepdims=True) + 1e-5
        
        
        return ((x - m)/T.sqrt(v))*self.gamma + self.beta
        
    def parameters(self):
        return [self.gamma, self.beta]
    
    
class Dropout(Module):
    def __init__(self, p_drop) -> None:
        self.p_keep = 1 - p_drop
    
    def __call__(self, x, training=True):
        if training:
            binary_mask = T.rand(x.shape) < self.p_keep
            result = x * binary_mask
            return result / self.p_keep
        return x
    
    def parameters(self):
        return []
    
    
if __name__=="__main__":
    l = LinearLayer(3, 1, nonlin=T.tanh)
    b = BatchNorm1D(1)
    ll = LayerNorm(1)
    d = Dropout(0.5)
    x = T.rand((10,3))
    o = ll(b(d(l(x))))
    o.backward()
    print(o.shape)