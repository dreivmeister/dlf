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
        
    def __rsub__(self, other):
        return (-self) + other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.true_divide(self.data, other.data), (self, other), op=self.__truediv__)
        
        def grad_fn(g): # g is gradient
            self.grad += unbroadcast(self.data, g / other.data)
            other.grad += unbroadcast(other.data, - g * self.data / other.data**2)
        out.grad_fn = grad_fn
        
        return out

    def __rtruediv__(self, other):
        return other / self
    
    
    # should be used with caution
    def __mod__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.mod(self.data, other.data), (self, other), op=self.__mod__)
        
        def grad_fn(g): # g is gradient
            self.grad += unbroadcast(self.data, g)
            other.grad += unbroadcast(other.data, -g * np.floor(self.data/other.data))
        out.grad_fn = grad_fn
        
        return out

    def __rmod__(self, other):
        return other % self
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.power(self.data, other.data), (self,), op=self.__pow__)
        
        def grad_fn(g):
            self.grad += unbroadcast(self.data, g * other.data * self.data ** np.where(other.data, other.data - 1, 1.))
            other.grad += unbroadcast(other.data, g * np.log(replace_zero(self.data, 1.)) * self.data ** other.data)
        out.grad_fn = grad_fn
        
        return out

    def __rpow__(self, other):
        return other ** self
    
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
    
    @staticmethod
    def reshape(x, shape, order=None):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.reshape(x.data, shape=shape, order=order), (x,), op=Tensor.reshape)

        def grad_fn(g):
            x.grad += np.reshape(g, np.shape(x.data), order=order)
        out.grad_fn = grad_fn

        return out

    
    @staticmethod
    def roll(x, shift, axis=None):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.roll(x.data, shift=shift, axis=axis), (x,), op=Tensor.roll)

        def grad_fn(g):
            x.grad += np.roll(g, -shift, axis=axis)
        out.grad_fn = grad_fn

        return out
    
    @staticmethod
    def array_split(x, idxs, axis=0):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.array_split(x.data, indices_or_sections=idxs, axis=axis), (x,), op=Tensor.array_split)

        def grad_fn(g):
            x.grad += np.concatenate(g, axis=axis)
        out.grad_fn = grad_fn

        return out
    
    @staticmethod
    def split(x, idxs, axis=0):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.split(x.data, indices_or_sections=idxs, axis=axis), (x,), op=Tensor.split)

        def grad_fn(g):
            x.grad += np.concatenate(g, axis=axis)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def vsplit(x, idxs):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.vsplit(x.data, indices_or_sections=idxs), (x,), op=Tensor.vsplit)

        def grad_fn(g):
            x.grad += np.concatenate(g, axis=0)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def hsplit(x, idxs):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.hsplit(x.data, indices_or_sections=idxs), (x,), op=Tensor.hsplit)

        def grad_fn(g):
            x.grad += np.concatenate(g, axis=1)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def dsplit(x, idxs):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.dsplit(x.data, indices_or_sections=idxs), (x,), op=Tensor.dsplit)

        def grad_fn(g):
            x.grad += np.concatenate(g, axis=2)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def ravel(x, order=None):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.ravel(x.data, order=order), (x,), op=Tensor.ravel)

        def grad_fn(g):
            x.grad += np.reshape(g, np.shape(x.data), order=order)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def expand_dims(x, axis):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.expand_dims(x.data, axis=axis), (x,), op=Tensor.expand_dims)

        def grad_fn(g):
            x.grad += np.reshape(g, np.shape(x.data))
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def squeeze(x, axis=None):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.squeeze(x.data, axis=axis), (x,), op=Tensor.squeeze)

        def grad_fn(g):
            x.grad += np.reshape(g, np.shape(x.data))
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def diag(x, k=0):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.diag(x.data, k=k), (x,), op=Tensor.diag)

        def grad_fn(g):
            x.grad += np.diag(g, k=k)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def flipud(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.flipud(x.data), (x,), op=Tensor.flipud)

        def grad_fn(g):
            x.grad += np.flipud(g)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def fliplr(x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.fliplr(x.data), (x,), op=Tensor.fliplr)

        def grad_fn(g):
            x.grad += np.fliplr(g)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def rot90(x, k=1):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.rot90(x.data, k=k), (x,), op=Tensor.rot90)

        def grad_fn(g):
            x.grad += np.rot90(g, k=-k)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def trace(x, offset=1):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.trace(x.data, offset=offset), (x,), op=Tensor.trace)

        def grad_fn(g):
            x.grad += np.einsum('ij,...->ij...', np.eye(x.data.shape[0], x.data.shape[1], k=offset), g)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def triu(x, k=0):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.triu(x.data, k=k), (x,), op=Tensor.triu)

        def grad_fn(g):
            x.grad += np.triu(g, k=k)
        out.grad_fn = grad_fn

        return out
    
    
    @staticmethod
    def tril(x, k=0):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.tril(x.data, k=k), (x,), op=Tensor.tril)

        def grad_fn(g):
            x.grad += np.tril(g, k=k)
        out.grad_fn = grad_fn

        return out
    
    
    
    
    
    
    
    
    

    """
    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.equal(self.data, other.data), (self, other), op=self.__eq__)

        def grad_fn(g):
            self.grad += g
            other.grad += g
        out.grad_fn = grad_fn

        return out
    """






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
