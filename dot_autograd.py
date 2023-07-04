
def dot_adjoint_0(B, G, A_meta, B_meta):
    _, A_ndim, A_dtype, _ = A_meta
    _, B_ndim, _, _ = B_meta
    
    if B_ndim == 0 or B_ndim == 1 or A_ndim == 0:
        contract_num = max(0, B_ndim - (A_ndim != 0))
        out = onp.tensordot(G, B, contract_num)
    else:
        out = onp.tensordot(G, onp.swapaxes(B, -1, -2), B_ndim - 1)
    return onp.asarray(out, dtype=A_dtype)

def dot_adjoint_1(A, G, A_meta, B_meta):
    _, A_ndim, _, _ = A_meta
    _, B_ndim, B_dtype, _ = B_meta
    
    needs_transpose = B_ndim > 1 and A_ndim != 0
    swap = (lambda x: onp.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)
    if A_ndim == 0 or A_ndim == 1 or B_ndim == 0:
        contract_num = max(0, A_ndim - (B_ndim != 0))
        out = swap(onp.tensordot(G, A, contract_num))
    else:
        out = swap(onp.tensordot(
            G, A, [range(-A_ndim - B_ndim + 2, -B_ndim + 1), range(A_ndim - 1)]))
    return onp.asarray(out, dtype=B_dtype)

def dot_vjp_0(ans, A, B):
    A_meta, B_meta = anp.metadata(A), anp.metadata(B)
    
    return lambda g: match_complex(A, dot_adjoint_0(B, g, A_meta, B_meta))

def dot_vjp_1(ans, A, B):
    A_meta, B_meta = anp.metadata(A), anp.metadata(B)
    return lambda g: match_complex(B, dot_adjoint_1(A, g, A_meta, B_meta))
defvjp(anp.dot, dot_vjp_0, dot_vjp_1)

defvjp(dot_adjoint_0, lambda ans, B, g, An, Bn: lambda A: match_complex(B, dot_adjoint_1(A, g, An, Bn)),
                      lambda ans, B, g, An, Bn: lambda A: match_complex(g, anp.dot(A, B)))

defvjp(dot_adjoint_1, lambda ans, A, g, An, Bn: lambda B: match_complex(A, dot_adjoint_0(B, g, An, Bn)),
                      lambda ans, A, g, An, Bn: lambda B: match_complex(g, anp.dot(A, B)))





