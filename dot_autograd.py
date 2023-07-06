def matmul_adjoint_0(B, G, A_meta, B_ndim):
    if anp.ndim(G) == 0:  # A_ndim == B_ndim == 1
        return unbroadcast(G * B, A_meta)
    _, A_ndim, _, _ = A_meta
    if A_ndim == 1:
        G = anp.expand_dims(G, anp.ndim(G) - 1)
    if B_ndim == 1:  # The result we need is an outer product
        B = anp.expand_dims(B, 0)
        G = anp.expand_dims(G, anp.ndim(G))
    else:  # We need to swap the last two axes of B
        B = anp.swapaxes(B, B_ndim - 2, B_ndim - 1)
    result = anp.matmul(G, B)
    return unbroadcast(result, A_meta)

def matmul_adjoint_1(A, G, A_ndim, B_meta):
    if anp.ndim(G) == 0:  # A_ndim == B_ndim == 1
        return unbroadcast(G * A, B_meta)
    _, B_ndim, _, _ = B_meta
    B_is_vec = (B_ndim == 1)
    if B_is_vec:
        G = anp.expand_dims(G, anp.ndim(G))
    if A_ndim == 1:  # The result we need is an outer product
        A = anp.expand_dims(A, 1)
        G = anp.expand_dims(G, anp.ndim(G) - 1)
    else:  # We need to swap the last two axes of A
        A = anp.swapaxes(A, A_ndim - 2, A_ndim - 1)
    result = anp.matmul(A, G)
    if B_is_vec:
        result = anp.squeeze(result, anp.ndim(G) - 1)
    return unbroadcast(result, B_meta)

def matmul_vjp_0(ans, A, B):
    A_meta = anp.metadata(A)
    B_ndim = anp.ndim(B)
    return lambda g: matmul_adjoint_0(B, g, A_meta, B_ndim)

def matmul_vjp_1(ans, A, B):
    A_ndim = anp.ndim(A)
    B_meta = anp.metadata(B)
    return lambda g: matmul_adjoint_1(A, g, A_ndim, B_meta)

defvjp(anp.matmul, matmul_vjp_0, matmul_vjp_1)