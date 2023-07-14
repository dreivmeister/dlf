from typing import Any
import numpy as np
from tensor import Tensor as T

def transpose_last_two(x):
    return T.transpose(x, axes=list(range(len(x.shape)-2))+[-1, -2])


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
        act = T.dot(x, self.w.transpose((1,0)))
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
    
    
class MaxPool2d(Module):
    def __init__(self, pool):
        self.pool = pool
        
    def __call__(self, x):
        new_shape = x.shape[:2]
        for i in [0, 1]:
            pool_w = self.pool[i]
            img_w = x.shape[i+2]
            new_shape += (img_w // pool_w, pool_w)
        result = T.reshape(x, new_shape)
        return T.max(T.max(result, axis=3), axis=4)
    
    def parameters(self):
        return []

    
    
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
    
# only stride 1 and valid
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size
        self.in_c = in_channels # input depth
        self.out_c = out_channels # num filters
        
        kernels_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.kernels = T.rand(kernels_shape)
        
    def __call__(self, x):
        batch_size, channels, height, width = x.shape
        out_shape = (batch_size, self.out_c, height-self.kernel_size+1, width-self.kernel_size+1)
        return T.conv2d(x, self.kernels, out_shape)
    

    
    
class AttentionHead(Module):
    def __init__(self, block_size, n_embd, head_size, dropout=0.2, mask=False):
        self.key = LinearLayer(n_embd, head_size, bias=False)
        self.query = LinearLayer(n_embd, head_size, bias=False)
        self.value = LinearLayer(n_embd, head_size, bias=False)
        self.do_mask = mask
        if mask:
            m = np.zeros((block_size,block_size))
            m[np.triu_indices(block_size,1)] = -np.inf
            self.mask = T(m)
        
        self.dropout = Dropout(dropout)
    
    def __call__(self, x):
        b, t, c = x.shape # (10,4,16) (batch_size,block_size,n_embd)
        k = self.key(x) # (batch_size,block_size,n_embd) @ (n_embd, head_size) 
        q = self.query(x) # (B,T,C)
        wei = T.matmul(q, T.transpose(k, (0,2,1))) # transpose last two dims
        if self.do_mask:
            wei = wei + self.mask
        wei = softmax(wei, axis=2) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = T.matmul(wei, v)
        return out
    
    def parameters(self):
        return [*self.key.parameters(),*self.query.parameters(),*self.value.parameters()]
    
class MHA(Module):
    def __init__(self, block_size, n_embd, num_heads, head_size, dropout=0.5, do_mask=False):
        self.heads = [AttentionHead(block_size=block_size,n_embd=n_embd,head_size=head_size,mask=do_mask) for _ in range(num_heads)]
        self.proj = LinearLayer(n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        out = T.concatenate([h(x) for h in self.heads],axis=-1)
        out = self.dropout(self.proj(out))
        return out

    def parameters(self):
        return [*self.proj.parameters()] + [p for head in self.heads for p in head.parameters()]
    
class FeedForward(Module):
    def __init__(self, n_embd):
        self.ll1 = LinearLayer(n_embd, 4*n_embd,nonlin=T.relu)
        self.ll2 = LinearLayer(4*n_embd, n_embd)
        self.drop = Dropout(0.5)
    
    def __call__(self, x):
        return self.drop(self.ll2(self.ll1(x)))
    
    def parameters(self):
        return [*self.ll1.parameters(), *self.ll2.parameters()]

class EncoderBlock(Module):
    def __init__(self, block_size, n_embd, num_heads, dropout=0.5, do_mask=False):
        # block_size - context_length - length of sample
        # n_embd - embedding_dimension - d_model
        # num_heads - number of heads in MHA
        # head_size - embedding dimension in single head
        head_size = n_embd // num_heads
        self.sa = MHA(block_size,n_embd,num_heads,head_size,dropout,do_mask)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    def parameters(self):
        return [*self.sa.parameters(),*self.ln1.parameters(),*self.ln2.parameters(),*self.ffwd.parameters()]
    
    
    
class VanillaRNNBlock(Module):
    
    def __init__(self, N, T, D, H, h0):
        """
        - batch size - N
        - seq length - T
        - elem dim   - D
        - hidden dim - H
        """
        
        self.N = N
        self.T = T
        self.H = H
        self.prev_h = h0 # previous hidden state is h0
        self.Wx = T.rand((D,H))
        self.Wh = T.rand((H,H))
        self.b = T.rand((H,))
    
    def rnn_step(self, x):
        # x is of shape (N,D)
        return T.tanh(T.dot(x, self.Wx) + T.dot(self.prev_h, self.Wh) + self.b)
    
    def __call__(self, x):
        # x is of shape (N,T,D)
        seq = []
        
        for i in range(self.T):
            step_out = self.rnn_step(x[:,i,:]) # step_out (N,H)
            seq.append(step_out)
            self.prev_h = step_out
        
        return T.reshape(T.concatenate(seq), (self.N,self.T,self.H))
    
    def parameters(self):
        return [self.Wx, self.Wh, self.b]
    
    
def softmax(logits, axis=1):
    logits = logits - T.max(logits, axis=axis, keepdims=True)
    logits_exp = T.exp(logits)
    exp_sum = T.sum(logits_exp, axis=axis, keepdims=True)
    return logits_exp / exp_sum
    
def negative_log_likelihood(probs, targets):
    # binary classification
    # preds is a probability vector
    # targets is a vector of zeros and ones
    label_probs = probs * targets + (1 - probs) * (1 - targets)
    return -(T.sum(T.log(label_probs)))

def cross_entropy(probs, targets):
    # preds is a probability vector (each column sums to one)
    # targets is a one hot vector
    log_probs = T.log(probs + 10e-20)
    return -T.sum(targets * log_probs)

def mse_loss(preds, targets, reduction='mean'):
    # preds is a prediction vector
    # target contains the target values
    if reduction == 'mean':
        return T.mean((preds - targets)**2)
    elif reduction == 'sum':
        return T.sum((preds - targets)**2)
    
    
if __name__=="__main__":
    # l = LinearLayer(3, 1, nonlin=T.tanh)
    # b = BatchNorm1D(1)
    # ll = LayerNorm(1)
    # d = Dropout(0.5)
    # x = T.rand((10,3))
    # o = ll(b(d(l(x))))
    # o.backward()
    
    # x = T.rand((10,4,16))
    # ah = MHA(4,16,4,mask=True)
    # o = ah(x)
    # print('o', o.shape)
    # o.backward()
    # print(o.shape)
    
    
    batch_size = 4
    block_size = 8

    n_embd = 32
    n_head = 4
    head_size = n_embd // n_head
    x = T.rand((batch_size,block_size,n_embd))
    
    B = EncoderBlock(block_size,n_embd,n_head,head_size,do_mask=True)
    o = B(x)
    o.backward()
    print(o.shape)
    
    
    # B = AttentionHead(block_size,n_embd,head_size,mask=True)
    # o = B(x)
    # print(o.shape)
    # o.backward()