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
    
    
    
