def conv2d(x, kernels, output_shape):
    # only stride=1 and valid padding
    #https://github.com/TheIndependentCode/Neural-Network/blob/master/convolutional.py
    # self is the input image
    # other is a kernel
    # forward
    
    batch_size, out_channels, out_height, out_width = output_shape
    out = np.random.randn(*output_shape)
    
    in_channels = x.shape[1]
    #out = np.zeros_like(*output_shape)
    for k in range(batch_size):
        for i in range(out_channels):
            for j in range(in_channels):
                out[k,i] += signal.correlate2d(x.data[k,j], kernels.data[i,j], "valid")
    
    out = Tensor(out, (x, kernels), op=Tensor.conv2d)
    def grad_fn(gradient):
        x.grad = np.zeros_like(x.data)
        kernels.grad = np.zeros_like(kernels.data)

        for k in range(batch_size):
            for i in range(out_channels):
                for j in range(in_channels):
                    kernels.grad[i,j] += signal.correlate2d(x.data[k,j], gradient[k,i], "valid")
                    x.grad[k,j] += signal.convolve2d(gradient[k,i], kernels.data[i,j], "full")
    out.grad_fn = grad_fn
    
    return out







class Conv2d(Module):
    # only valid only stride 1
    def __init__(self, in_channels, out_channels, kernel_size):
        # input_shape - shape of input image (batch_size, channel_dim, height, width)
        # kernel_size - square kernel size only, int
        # depth - num of kernels, num of channels in output image
        #batch_size, in_channels, input_height, input_width = input_shape
        self.kernel_size = kernel_size
        #self.num_filters = out_channels # num of kernels
        #self.input_shape = input_shape
        self.input_depth = in_channels
        self.num_filters = out_channels
        #self.output_shape = (batch_size, num_filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (out_channels, in_channels, kernel_size, kernel_size)
        
        self.kernels = Tensor.rand(self.kernels_shape)
        #self.kernels = Tensor(np.random.randn(*self.kernels_shape))
    def __call__(self, x):
        # x is a Tensor of shape (batch_size, channel_dim, height, width)
        #out = Tensor(np.copy(self.bias))
        output_shape = (x.shape[0],self.num_filters,x.shape[2]-self.kernel_size+1,x.shape[3]-self.kernel_size+1)
        return x.conv2d(self.kernels, output_shape)
    
    def parameters(self):
        return [self.kernels]
    
    
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
        return conv2d(x, self.kernels, out_shape)