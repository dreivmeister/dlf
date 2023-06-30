# basically everthing you need

import numpy as np

def unbroadcast(target, g, broadcast_idx=0):
    """Remove broadcasted dimensions by summing along them.

    When computing gradients of a broadcasted value, this is the right thing to
    do when computing the total derivative and accounting for cloning.
    """
    while np.ndim(g) > np.ndim(target):
        g = np.sum(g, axis=broadcast_idx)
    for axis, size in enumerate(np.shape(target)):
        if size == 1:
            g = np.sum(g, axis=axis, keepdims=True)
    if np.iscomplexobj(g) and not np.iscomplex(target):
        g = np.real(g)
    return g

def replace_zero(x, val):
    """Replace all zeros in 'x' with 'val'."""
    return np.where(x, x, val)


class Tensor:
    def __init__(self, data, prev=(), op=lambda x: None, name=None, *args, **kwargs) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.prev = prev
        self.grad = 0
        self.op = op
        self.grad_fn = lambda x: None
        self.broadcast_dim = None # maybe no need
        self.name = name
        
    def __repr__(self):
        if self.data.ndim < 2:
            return f'Tensor(data={self.data}, grad={self.grad})'    
        return f'Tensor\ndata=\n{self.data},\ngrad=\n{self.grad})'
    
    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data, dtype=np.float32)
        self.grad = gradient

        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t.prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        for t in reversed(topo):
            t.grad_fn(t.grad)
            
    # operations
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.add(self.data, other.data), (self, other), op=self.__add__)
        
        def grad_fn(g): # g is gradient
            self.grad += unbroadcast(self.data, g)
            other.grad += unbroadcast(other.data, g)
        out.grad_fn = grad_fn
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.multiply(self.data, other.data), (self, other), op=self.__mul__)
        
        def grad_fn(g):
            self.grad += unbroadcast(self.data, other.data * g)
            other.grad += unbroadcast(other.data, self.data * g)
        out.grad_fn = grad_fn
        
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.subtract(self.data, other.data), (self, other), op=self.__sub__)
        
        def grad_fn(g): # g is gradient
            self.grad += unbroadcast(self.data, g)
            other.grad += unbroadcast(other.data, -g)
        out.grad_fn = grad_fn
        
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.true_divide(self.data, other.data), (self, other), op=self.__truediv__)
        
        def grad_fn(g): # g is gradient
            self.grad += unbroadcast(self.data, g / other.data)
            other.grad += unbroadcast(other.data, - g * self.data / other.data**2)
        out.grad_fn = grad_fn
        
        return out
    
    def __mod__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.mod(self.data, other.data), (self, other), op=self.__mod__)
        
        def grad_fn(g): # g is gradient
            self.grad += unbroadcast(self.data, g)
            other.grad += unbroadcast(other.data, -g * np.floor(self.data/other.data))
        out.grad_fn = grad_fn
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        other = Tensor(other)
        out = Tensor(np.power(self.data, other.data), (self,), op=self.__pow__)
        
        def grad_fn(g):
            self.grad += unbroadcast(self.data, g * other.data * self.data ** np.where(other.data, other.data - 1, 1.))
            other.grad += unbroadcast(other.data, g * np.log(replace_zero(self.data, 1.)) * self.data ** other.data)
        out.grad_fn = grad_fn
        
        return out
    
    
    







if __name__=="__main__":
    pass