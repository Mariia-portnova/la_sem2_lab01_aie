# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"forms: {tt1.shape} != {tt2.shape}")

    d = tt1.order
    new_cores = []

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        r1_left, n_k, r1_right = core1.shape
        r2_left, n_k2, r2_right = core2.shape

        if n_k != n_k2:
            raise ValueError(f"size mod {k} not same")

        if k == 0:
            new_shape = (1, n_k, r1_right + r2_right)
            new_core = DenseTensor.zeros(new_shape)
            for i in range(n_k):
                for j in range(r1_right):
                    new_core[0, i, j] = core1[0, i, j]
                for j in range(r2_right):
                    new_core[0, i, r1_right + j] = core2[0, i, j]

        elif k == d - 1:
            new_shape = (r1_left + r2_left, n_k, 1)
            new_core = DenseTensor.zeros(new_shape)
            for i in range(r1_left):
                for j in range(n_k):
                    new_core[i, j, 0] = core1[i, j, 0]
            for i in range(r2_left):
                for j in range(n_k):
                    new_core[r1_left + i, j, 0] = core2[i, j, 0]

        else:
            new_shape = (r1_left + r2_left, n_k, r1_right + r2_right)
            new_core = DenseTensor.zeros(new_shape)
            for i in range(r1_left):
                for j in range(n_k):
                    for p in range(r1_right):
                        new_core[i, j, p] = core1[i, j, p]
            for i in range(r2_left):
                for j in range(n_k):
                    for p in range(r2_right):
                        new_core[r1_left + i, j, r1_right + p] = core2[i, j, p]

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    new_cores = [tt.cores[0].copy() for _ in range(tt.order)]

    r_left, n_0, r_right = new_cores[0].shape
    for i in range(r_left):
        for j in range(n_0):
            for p in range(r_right):
                new_cores[0][i, j, p] = alpha * tt.cores[0][i, j, p]

    for k in range(1, tt.order):
        new_cores[k] = tt.cores[k].copy()

    return TTTensor(new_cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"forms: {tt1.shape} != {tt2.shape}")

    d = tt1.order
    new_cores = []

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        r1_left, n_k, r1_right = core1.shape
        r2_left, n_k2, r2_right = core2.shape

        if n_k != n_k2:
            raise ValueError(f"size mod {k} nor right")

        new_shape = (r1_left * r2_left, n_k, r1_right * r2_right)
        new_core = DenseTensor.zeros(new_shape)

        for i1 in range(r1_left):
            for i2 in range(r2_left):
                for j in range(n_k):
                    for p1 in range(r1_right):
                        for p2 in range(r2_right):
                            new_i = i1 * r2_left + i2
                            new_p = p1 * r2_right + p2
                            new_core[new_i, j, new_p] = core1[i1, j, p1] * core2[i2, j, p2]

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"{tt1.shape} != {tt2.shape}")

    d = tt1.order
    Z = backend.eye(1)

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        n_k = core1.shape[1]
        
        Z_new = DenseTensor.zeros((Z.shape[0] * core1.shape[0], Z.shape[1] * core2.shape[2]))
        
        for idx in range(n_k):
            slice1 = DenseTensor.zeros((core1.shape[0], core1.shape[2]))
            slice2 = DenseTensor.zeros((core2.shape[0], core2.shape[2]))
            
            for i in range(core1.shape[0]):
                for j in range(core1.shape[2]):
                    slice1[i, j] = core1[i, idx, j]
            for i in range(core2.shape[0]):
                for j in range(core2.shape[2]):
                    slice2[i, j] = core2[i, idx, j]
            
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    for p in range(core1.shape[0]):
                        for q in range(core2.shape[2]):
                            Z_new[i * core1.shape[0] + p, j * core2.shape[2] + q] += Z[i, j] * slice1[p, idx] * slice2[idx, q]
        
        Z = Z_new

    if Z.shape != (1, 1):
        raise ValueError("wrong result shape")

    return Z[0, 0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    dot_product = tt_dot(tt, tt, backend)
    return math.sqrt(dot_product)


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"{tt1.shape} != {tt2.shape}")

    norm1_sq = tt_dot(tt1, tt1, backend)
    norm2_sq = tt_dot(tt2, tt2, backend)
    dot12 = tt_dot(tt1, tt2, backend)

    diff_norm_sq = norm1_sq + norm2_sq - 2 * dot12

    if diff_norm_sq < 0:
        diff_norm_sq = 0.0

    return math.sqrt(diff_norm_sq)