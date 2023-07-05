


@primitive
def untake(x, idx, vs):
    if isinstance(idx, list) and (len(idx) == 0 or not isinstance(idx[0], slice)):
        idx = onp.array(idx, dtype='int64')
    def mut_add(A):
        onp.add.at(A, idx, x)
        return A
    return SparseObject(vs, mut_add)
defvjp(func(ArrayBox.__getitem__), lambda ans, A, idx: lambda g: untake(g, idx, vspace(A)))
defvjp(untake, lambda ans, x, idx, _: lambda g: g[idx])



def untake(x, idx, vs):
    if isinstance(idx, list) and (len(idx) == 0 or not isinstance(idx[0], slice)):
        idx = onp.array(idx, dtype='int64')
    def mut_add(A):
        onp.add.at(A, idx, x)
        return A
    return SparseObject(vs, mut_add)
    



def __getitem__(self, idx):
    out = Tensor(self.data[idx], (self,), op=self.__getitem__)
    
    def grad_fn(g):
        self.grad += untake(g, idx, vspace(A))