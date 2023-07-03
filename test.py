import autograd.numpy as np
from autograd import elementwise_grad


a = np.array([[1,2,3],[3,2,4]])
b = np.array([[5,2,3],[4,2,2]])


def f(a, b):
    if (a+1 <= b).any():
        return np.sin(a)


df1 = elementwise_grad(f,0)
df2 = elementwise_grad(f,1)
#print(f(a,b))
print(df1(a,b))
print(df2(a,b))