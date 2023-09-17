from linpyk.graph import DTArray, Empty, ops


def cholesky(a: DTArray) -> DTArray:
    if len(a.shape) != 2:
        raise ValueError("2D-array required.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Squared matrix required.")

    p = a.shape[0]
    if p < 1:
        raise ValueError(f"Invalid matrix shape {a.shape}.")

    l = Empty(a.shape, "L")
    a = [a] + [Empty((p - k, p - k), f"A{k}") for k in range(1, p)]

    for k in range(p):
        # a[k][k,k] = l[k,k] * l[k,k]**T
        l[k, k] = ops.tile_potrf(a=a[k][0, 0], lower=1, overwrite_a=0)
        for i in range(k + 1, p):
            # l[i,k] = a[k][i,k]*l[k,k]**-T
            l[i, k] = ops.tile_trsm(
                alpha=1.0, a=l[k, k], b=a[k][i - k, 0], side=1, lower=1, trans_a=1
            )
        for j in range(k + 1, p):
            for i in range(j, p):
                if i == j:
                    a[k + 1][i - k - 1, j - k - 1] = ops.tile_syrk(
                        alpha=-1.0, a=l[i, k], c=a[k][i - k, j - k], beta=1.0, lower=1
                    )
                else:
                    # a[k + 1][i,j] = a[k][i,j] - l[i,k]*l[j,k]**T
                    a[k + 1][i - k - 1, j - k - 1] = ops.tile_gemm(
                        alpha=-1.0,
                        a=l[i, k],
                        b=l[j, k],
                        beta=1.0,
                        c=a[k][i - k, j - k],
                        trans_b=1,
                    )

    return l


def forward_substitution(a: DTArray, b: DTArray, trans: bool = False, lower=False, unit_diagonal=False, label: str = "X"):
    p = a.shape[0]
    x = Empty(b.shape, label)
    b = [b] + [Empty((p - k, 1), f"{b._label}{k}") for k in range(1, p)]

    for j in range(p):
        x[j, 0] = ops.tile_trsm(alpha=1., a=a[j, j], b=b[j][0, 0], lower=int(lower), trans_a=int(trans), diag=unit_diagonal)
        for i in range(0, p - j - 1):
            #   b[i] = b[i] - A[i,j] * X[j]
            b[j + 1][i, 0] = ops.tile_gemm(alpha=-1., a=a[j + i + 1, j], b=x[j, 0], beta=1., c=b[j][i + 1, 0], trans_a=int(trans))

    return x


def backward_substitution(a: DTArray, b: DTArray, trans: bool = False, lower=False, unit_diagonal=False, label: str = "X"):
    p = a.shape[0]
    x = Empty(b.shape, label)
    b = [b] + [Empty((p - k, 1), f"{b._label}{k}") for k in range(1, p)]

    def swap(i, j):
        if trans:
            return j, i
        return i, j

    for j in reversed(range(p)):
        x[j, 0] = ops.tile_trsm(alpha=1., a=a[j, j], b=b[p - 1 - j][j, 0], lower=int(lower), trans_a=int(trans),
                                diag=unit_diagonal)
        for i in reversed(range(j)):
            #   b[i] = b[i] - A[i,j] * X[j]
            b[p - j][i, 0] = ops.tile_gemm(alpha=-1., a=a[swap(i, j)], b=x[j, 0], beta=1., c=b[p - 1 - j][i, 0],
                                           trans_a=int(trans))

    return x


def solve(a: DTArray, b: DTArray, assume_a: str = "gen") -> DTArray:
    match assume_a:
        case 'spd':
            l = cholesky(a)
            y = forward_substitution(l, b, lower=True, label="y")
            return backward_substitution(l, y, trans=True, lower=True, label="x")
        case _:
            raise NotImplementedError()
