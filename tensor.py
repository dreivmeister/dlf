# basically everthing you need

import numpy as np


# from autodidact
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
    
    
    # should be used with caution
    def __mod__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.mod(self.data, other.data), (self, other), op=self.__mod__)
        
        def grad_fn(g): # g is gradient
            self.grad += unbroadcast(self.data, g)
            other.grad += unbroadcast(other.data, -g * np.floor(self.data/other.data))
        out.grad_fn = grad_fn
        
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.power(self.data, other.data), (self,), op=self.__pow__)
        
        def grad_fn(g):
            self.grad += unbroadcast(self.data, g * other.data * self.data ** np.where(other.data, other.data - 1, 1.))
            other.grad += unbroadcast(other.data, g * np.log(replace_zero(self.data, 1.)) * self.data ** other.data)
        out.grad_fn = grad_fn
        
        return out
    
    def __neg__(self):
        out = Tensor(np.negative(self.data), (self,), op=self.__neg__)
        
        def grad_fn(g):
            self.grad += -g
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def sin(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.sin(x.data), (x,), op=Tensor.sin)
        
        def grad_fn(g):
            x.grad += g * np.cos(x.data)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def cos(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.cos(x.data), (x,), op=Tensor.cos)
        
        def grad_fn(g):
            x.grad += g * -np.sin(x.data)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def tan(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.tan(x.data), (x,), op=Tensor.tan)
        
        def grad_fn(g):
            x.grad += g / np.cos(x.data) ** 2
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def arcsin(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.arcsin(x.data), (x,), op=Tensor.arcsin)
        
        def grad_fn(g):
            x.grad += g / np.sqrt(1 - x.data**2)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def arccos(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.arccos(x.data), (x,), op=Tensor.arccos)
        
        def grad_fn(g):
            x.grad += - g / np.sqrt(1 - x.data**2)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def arctan(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.arctan(x.data), (x,), op=Tensor.arctan)
        
        def grad_fn(g):
            x.grad += - g / (1 + x.data**2)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def abs(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        # not really abs cause it only handles reals
        out = Tensor(np.fabs(x.data), (x,), op=Tensor.abs)
        
        def grad_fn(g):
            x.grad += g * np.sign(x.data)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def exp(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.exp(x.data), (x,), op=Tensor.exp)
        
        def grad_fn(g):
            x.grad += g * out.data
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def log(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.log(x.data), (x,), op=Tensor.log)
        
        def grad_fn(g):
            x.grad += g / x.data
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def tanh(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.tanh(x.data), (x,), op=Tensor.tanh)
        
        def grad_fn(g):
            x.grad += g / np.cosh(x.data) ** 2
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def sinh(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.sinh(x.data), (x,), op=Tensor.sinh)
        
        def grad_fn(g):
            x.grad += g * np.cosh(x.data)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def cosh(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.cosh(x.data), (x,), op=Tensor.cosh)
        
        def grad_fn(g):
            x.grad += g * np.sinh(x.data)
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def sqrt(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.sqrt(x.data), (x,), op=Tensor.sqrt)
        
        def grad_fn(g):
            x.grad += g / (2*out.data)
        out.grad_fn = grad_fn
        
        return out
    

    # def __eq__(self, other):
    #     other = other if isinstance(other, Tensor) else Tensor(other)
    #     out = Tensor(np.equal(self.data, other.data), (self, other), op=self.__eq__)
        
    #     def grad_fn(g):
    #         self.grad += g
    #         other.grad += g
    #     out.grad_fn = grad_fn
        
    #     return out






if __name__=="__main__":
    a = Tensor([[1,2,3],[3,4,5]])
    b = Tensor([[1,8,1],[8,3,5]])
    
    
    def f(a, b):
        #return a % b
        #return a - b * b ** a / a - b * 3 + 2
        return Tensor.sin(a+b)
    
    c = f(a, b)
    print(c)    
    c.backward()
    
    eps = 1e-3
    print(a)
    print((f(a.data + eps, b.data) - f(a.data, b.data))/eps)
    print(b)
    print((f(a.data, b.data + eps) - f(a.data, b.data))/eps)
    # doesnt match in last element