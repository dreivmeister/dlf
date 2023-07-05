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

def repeat_to_match_shape(g, shape, axis, keepdims):
    """Returns the array g repeated along axis to fit shape
       Also returns the number of repetitions of the array."""
    if shape == ():
      return g, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
    return np.broadcast_to(np.reshape(g, new_shape), shape), num_reps
    
def grad_chooser(g, ans, x, axis=None, keepdims=None):
    shape = np.shape(x)
    g_repeated, _ = repeat_to_match_shape(g, shape, axis, keepdims)
    argmax_locations = x == repeat_to_match_shape(ans, shape, axis, keepdims)[0]
    return g_repeated * argmax_locations / np.sum(argmax_locations, axis=axis, keepdims=True)


class Tensor:
    def __init__(self, data, prev=(), op=lambda x: None, name=None, *args, **kwargs) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.prev = prev
        self.grad = 0
        self.op = op
        self.grad_fn = lambda x: None
        self.name = name
    
    def __hash__(self):
        return id(self)
    
    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    
    @staticmethod
    def zeros(shape):
        return Tensor(np.zeros(shape))
    
    @staticmethod
    def zeros_like(tensor):
        return Tensor(np.zeros_like(tensor))
    
    @staticmethod
    def ones(shape):
        return Tensor(np.ones(shape))
    
    @staticmethod
    def ones_like(tensor):
        return Tensor(np.ones_like(tensor))
    
    @staticmethod
    def eye(dim):
        return Tensor(np.eye(dim))
    
    @staticmethod
    def rand(shape):
        return Tensor(np.random.rand(*shape))
    
    @staticmethod
    def uniform(low, high, shape):
        return Tensor(np.random.uniform(low,high,shape))
    
    def any(self):
        out = Tensor(np.any(self.data), (self,), op=self.__le__)
        return out
    def all(self):
        out = Tensor(np.all(self.data), (self,), op=self.__le__)
        return out
        
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
            
            
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), op=self.__getitem__)
        
        def grad_fn(g):
            self_prime = np.zeros_like(self.data)
            self_prime[idx] = g
            self.grad += self_prime
        out.grad_fn = grad_fn
    
        return out
    
    
    # relativley sure about these
    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.less_equal(self.data, other.data), (self, other), op=self.__le__)
        return out
    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.less(self.data, other.data), (self, other), op=self.__le__)
        return out
    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.greater_equal(self.data, other.data), (self, other), op=self.__le__)
        return out
    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.greater(self.data, other.data), (self, other), op=self.__le__)
        return out
    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.equal(self.data, other.data), (self, other), op=self.__le__)
        return out        
            
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
    
    
    @staticmethod
    def clip(x, a_min, a_max):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.clip(x.data, a_min=a_min, a_max=a_max), (x,), op=Tensor.clip)
        
        def grad_fn(g):
            x.grad += g * np.logical_and(out.data != a_min, out.data != a_max)
        out.grad_fn = grad_fn
        
        return out
    
    
    @staticmethod
    def swapaxes(x, axis1, axis2):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.swapaxes(x.data, axis1=axis1, axis2=axis2), (x,), op=Tensor.swapaxes)
        
        def grad_fn(g):
            x.grad += np.swapaxes(g, axis2, axis1)
        out.grad_fn = grad_fn
        
        return out
    
    
    @staticmethod
    def moveaxis(x, source, destination):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.moveaxis(x.data, source=source, destination=destination), (x,), op=Tensor.moveaxis)
        
        def grad_fn(g):
            x.grad += np.moveaxis(g, destination, source)
        out.grad_fn = grad_fn
        
        return out
    
    
    @staticmethod
    def repeat(x, repeats, axis=None):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.repeat(x.data, repeats=repeats, axis=axis), (x,), op=Tensor.repeat)
        shape = np.shape(x.data)
        
        def grad_fn(g):
            if axis is None:  # If axis is none, np.repeat() repeats the flattened array.
                expanded = np.reshape(g, (np.prod(shape),) + (repeats,))
                x.grad += np.reshape(np.sum(expanded, axis=1, keepdims=False), shape)
            else:
                if shape[axis] == 1:  # For this common case, the logic is simple.
                    x.grad += np.sum(g, axis=axis, keepdims=True)
                else:
                    expanded = np.reshape(g, shape[0:axis+1] + (repeats,) + shape[axis+1:])
                    x.grad += np.sum(expanded, axis=axis+1, keepdims=False)
        out.grad_fn = grad_fn
        
        return out
    
    
    # might work
    @staticmethod
    def concatenate(a, axis=0):
        a = [x if isinstance(x, Tensor) else Tensor(x) for x in a]
        out = Tensor(np.concatenate([x.data for x in a], axis=axis), (*a,), op=Tensor.concatenate)
        def grad_fn(g):
            lens = [x.data.shape[axis] for x in a]
            g_split = np.split(g, np.cumsum(lens), axis=axis)
            for i,x in enumerate(a):
                x.grad += g_split[i]    
        out.grad_fn = grad_fn
        
        return out
                
                
    @staticmethod
    def tile(x, reps):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.tile(x.data, reps=reps), (x,), op=Tensor.tile)
        reps = [reps] if np.isscalar(reps) else reps
        x_shape = np.shape(x.data)
        
        def grad_fn(g):
            for axis, rep in enumerate(reps):
                g = np.sum(np.split(g, rep, axis))
            x.grad += np.reshape(g, x_shape)
        out.grad_fn = grad_fn
        
        return out
        
    
    @staticmethod
    def transpose(x, axes=None):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.transpose(x.data, axes=axes), (x,), op=Tensor.transpose)
        
        if axes is not None:
            axes = np.argsort(axes)
        
        def grad_fn(g):
            x.grad += np.transpose(g, axes)
        out.grad_fn = grad_fn
        
        return out


    @staticmethod
    def broadcast_to(x, new_shape):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.broadcast_to(x.data, shape=new_shape), (x,), op=Tensor.broadcast_to)
        old_shape = np.shape(x.data)
        assert np.shape(out.data) == new_shape
        assert len(old_shape) == len(new_shape), "Can't handle extra leading dims"
        
        def grad_fn(g):
            broadcast_axes = tuple(np.where(np.logical_and(np.array(old_shape) == 1, np.array(new_shape) >  1))[0])
            x.grad += np.sum(g, axis=broadcast_axes, keepdims=True)
        out.grad_fn = grad_fn
        
        return out
    

    @staticmethod
    def sum(x, axis=None, keepdims=False):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.sum(x.data, axis=axis, keepdims=keepdims), (x,), op=Tensor.sum)
        x_shape = np.shape(x.data)
        
        def grad_fn(g):
            x.grad += repeat_to_match_shape(g, x_shape, axis, keepdims)[0]
        out.grad_fn = grad_fn
        
        return out
    
    @staticmethod
    def mean(x, axis=None, keepdims=False):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.mean(x.data, axis=axis, keepdims=keepdims), (x,), op=Tensor.mean)
        shape, dtype = np.shape(x.data), x.data.dtype
        
        def grad_fn(g):
            g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
            return g_repeated / num_reps
        out.grad_fn = grad_fn
        
        return out
    
    
    @staticmethod
    def prod(x, axis=None, keepdims=False):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.prod(x.data, axis=axis, keepdims=keepdims), (x,), op=Tensor.prod)
        shape, dtype = np.shape(x.data), x.data.dtype
        
        def grad_fn(g):
            g_repeated, _ = repeat_to_match_shape(g * out.data, shape, dtype, axis, keepdims)
            return g_repeated / x.data
        out.grad_fn = grad_fn
        
        return out


    @staticmethod
    def var(x, axis=None, ddof=0, keepdims=False):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.var(x.data, axis=axis, ddof=ddof, keepdims=keepdims), (x,), op=Tensor.var)
        shape, dtype = np.shape(x.data), x.data.dtype
        
        def grad_fn(g):
            g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
            x_minus_mean = x.data - np.mean(x.data, axis=axis, keepdims=True)
            return 2.0 * g_repeated * x_minus_mean / (num_reps - ddof)
        out.grad_fn = grad_fn
        
        return out

    @staticmethod
    def std(x, axis=None, ddof=0, keepdims=False):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.var(x.data, axis=axis, ddof=ddof, keepdims=keepdims), (x,), op=Tensor.var)
        shape, dtype = np.shape(x.data), x.data.dtype
        
        def grad_fn(g):
            g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)  # Avoid division by zero.
            if num_reps <= 1:
                return g_repeated * 0.0
            else:
                g_repeated, num_reps = repeat_to_match_shape(g / out.data, shape, dtype, axis, keepdims)
                x_minus_mean = x.data - np.mean(x.data, axis=axis, keepdims=True)
                return g_repeated * x_minus_mean / (num_reps - ddof)
        out.grad_fn = grad_fn
        
        return out


    @staticmethod
    def max(x, axis=None, keepdims=False):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.max(x.data, axis=axis, keepdims=keepdims), (x,), op=Tensor.max)
        
        def grad_fn(g):
            x.grad += grad_chooser(g, out.data, x.data, axis=axis, keepdims=keepdims)
        out.grad_fn = grad_fn
        
        return out
    
    
    @staticmethod
    def min(x, axis=None, keepdims=False):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(np.min(x.data, axis=axis, keepdims=keepdims), (x,), op=Tensor.min)
        
        def grad_fn(g):
            x.grad += grad_chooser(g, out.data, x.data, axis=axis, keepdims=keepdims)
        out.grad_fn = grad_fn
        
        return out
    
    
    #TODO: convolve
        
    @staticmethod
    def dot(a, b):
        a = a if isinstance(a, Tensor) else Tensor(a)
        b = b if isinstance(b, Tensor) else Tensor(b)
        out = Tensor(np.dot(a.data, b.data), (a,b), op=Tensor.dot)
        
        a_ndim = a.data.ndim
        b_ndim = b.data.ndim
        a_dtype = a.data.dtype
        b_dtype = b.data.dtype
        
        def grad_fn(g):
            if b_ndim == 0 or b_ndim == 1 or a_ndim == 0:
                contract_num = max(0, b_ndim - (a_ndim != 0))
                out = np.tensordot(g, b.data, contract_num)
            else:
                out = np.tensordot(g, np.swapaxes(b.data, -1, -2), b_ndim - 1)
            a.grad += np.asarray(out, dtype=a_dtype)
            
            
            needs_transpose = b_ndim > 1 and a_ndim != 0
            swap = (lambda x: np.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)
            if a_ndim == 0 or a_ndim == 1 or b_ndim == 0:
                contract_num = max(0, a_ndim - (b_ndim != 0))
                out = swap(np.tensordot(g, a.data, contract_num))
            else:
                out = swap(np.tensordot(
                    g, a.data, [range(-a_ndim - b_ndim + 2, -b_ndim + 1), range(a_ndim - 1)]))
            b.grad += np.asarray(out, dtype=b_dtype)
        out.grad_fn = grad_fn
        
        return out
    
def dot(a, b):
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = Tensor(np.dot(a.data, b.data), (a,b), op=Tensor.dot)
    
    a_ndim = a.data.ndim
    b_ndim = b.data.ndim
    a_dtype = a.data.dtype
    b_dtype = b.data.dtype
    
    def grad_fn(g):
        if b_ndim == 0 or b_ndim == 1 or a_ndim == 0:
            contract_num = max(0, b_ndim - (a_ndim != 0))
            out = np.tensordot(g, b.data, contract_num)
        else:
            out = np.tensordot(g, np.swapaxes(b.data, -1, -2), b_ndim - 1)
        a.grad += np.asarray(out, dtype=a_dtype)
        
        
        needs_transpose = b_ndim > 1 and a_ndim != 0
        swap = (lambda x: np.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)
        if a_ndim == 0 or a_ndim == 1 or b_ndim == 0:
            contract_num = max(0, a_ndim - (b_ndim != 0))
            out = swap(np.tensordot(g, a.data, contract_num))
        else:
            out = swap(np.tensordot(
                g, a.data, [range(-a_ndim - b_ndim + 2, -b_ndim + 1), range(a_ndim - 1)]))
        b.grad += np.asarray(out, dtype=b_dtype)
    out.grad_fn = grad_fn
    
    return out
    


if __name__=="__main__":
    # a = Tensor([[1,2,3],[3,4,5],[3,4,5]])
    # b = Tensor([[1,8,1],[8,3,5],[8,3,5]])
    
    
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6]])
    
    
    c = Tensor.concatenate((a,Tensor.transpose(b)),axis=1)
    print(c)
    c.backward()
    
    print(a)
    print(b)
    
    
    
    
    # def f(a, b):
    #     #  return a % b
    #     # return a - b * b ** a / a - b * 3 + 2
    #     # if (a+1 <= b).any():
    #     #     return Tensor.sin(a+b)
    #     return Tensor.sin(Tensor.cos(a)[:2,:1])
    #     #return Tensor.dot(a,Tensor.max(b))
    
    # c = f(a, b)
    # print(c)    
    # c.backward()
    
    # print(a)
    # print(b)
    
    # eps = 1e-3
    # print(a)
    # print((f(a.data + eps, b.data) - f(a.data, b.data))/eps)
    # print(b)
    # print((f(a.data, b.data + eps) - f(a.data, b.data))/eps)
    # doesnt match in last element
