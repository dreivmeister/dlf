import numpy as np
from tensor import Tensor as T

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
        act = x @ self.w.transpose((-1,-2))
        if self.bias:
            act = act + self.b
        return getattr(act, self.nonlin)() if self.nonlin else act
    
    def parameters(self):
        if self.bias:
            return [self.w, self.b]
        return [self.w]