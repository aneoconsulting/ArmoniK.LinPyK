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
