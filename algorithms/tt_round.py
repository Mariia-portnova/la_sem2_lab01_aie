# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    tt_right = right_canonicalize(tt, backend)
    d = tt_right.order
    
    norm = tt_right.cores[0].norm()
    delta = eps * norm / math.sqrt(d - 1) if d > 1 else 0.0

    cores = []
    current = None  

    for k in range(d - 1):
        core = tt_right.cores[k].copy()
        
        r_left, n_k, r_right = core.shape
        
        if current is not None:
            new_core_data = []
            for i in range(current.shape[0]):
                for idx in range(n_k):
                    for p in range(r_right):
                        val = 0.0
                        for j in range(r_left):
                            val += current[i, j] * core[j, idx, p]
                        new_core_data.append(val)
            core = DenseTensor((current.shape[0], n_k, r_right), data=new_core_data)
            r_left = current.shape[0]

        matrix = core.reshape((r_left * n_k, r_right))
        
        U, S, Vt = backend.svd(matrix)
        
        rank = _compute_rank(S, delta, max_rank)
        if rank < 1:
            rank = 1
        
        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)
        
        new_core = U_trunc.reshape((r_left, n_k, rank))
        cores.append(new_core)
        
        current = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

    last_core = tt_right.cores[-1].copy()
    if current is not None:
        r_left, n_last, r_right = last_core.shape
        new_last_data = []
        for i in range(current.shape[0]):
            for idx in range(n_last):
                for p in range(r_right):
                    val = 0.0
                    for j in range(r_left):
                        val += current[i, j] * last_core[j, idx, p]
                    new_last_data.append(val)
        last_core = DenseTensor((current.shape[0], n_last, r_right), data=new_last_data)

    cores.append(last_core)
    return TTTensor(cores)

# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    k = S.size
    squared_sum = 0.0
    total_squared_sum = 0.0

    for i in range(k):
        total_squared_sum += S[i] * S[i]


    rank = k
    for r in range(k):
        if r < k:
            squared_sum += S[r] * S[r]
        remaining_squared = total_squared_sum - squared_sum
        if remaining_squared <= delta * delta:
            rank = r + 1
            break

    if max_rank is not None and rank > max_rank:
        rank = max_rank

    if rank < 1:
        rank = 1

    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError(f"_truncate_columns expected 2D, got {matrix.ndim}D")

    m, n = matrix.shape
    if rank < 0 or rank > n:
        raise ValueError(f"rank {rank} out of [0, {n}]")

    if rank == 0:
        return DenseTensor.zeros((m, 0))

    result_data = []
    for i in range(m):
        for j in range(rank):
            result_data.append(matrix[i, j])

    return DenseTensor((m, rank), data=result_data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError(f"_truncate_rows expected 2D, got {matrix.ndim}D")

    k, n = matrix.shape
    if rank < 0 or rank > k:
        raise ValueError(f"rank {rank} out of [0, {k}]")

    if rank == 0:
        return DenseTensor.zeros((0, n))

    result_data = []
    for i in range(rank):
        for j in range(n):
            result_data.append(matrix[i, j])

    return DenseTensor((rank, n), data=result_data)


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError(f"_truncate_vector expected 1D, got {vector.ndim}D")

    if rank < 0 or rank > vector.size:
        raise ValueError(f"rank {rank} out o [0, {vector.size}]")

    if rank == 0:
        return DenseTensor((0,), data=[])

    result_data = [vector[i] for i in range(rank)]
    return DenseTensor((rank,), data=result_data)


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1:
        raise ValueError(f"_multiply_diag_matrix expected 1D для diag_vec, got {diag_vec.ndim}D")
    if matrix.ndim != 2:
        raise ValueError(f"_multiply_diag_matrix expected 2D для matrix, got {matrix.ndim}D")

    m, n = matrix.shape
    if m != rank or diag_vec.size != rank:
        raise ValueError("wrong sizes")

    result_data = []
    for i in range(rank):
        for j in range(n):
            result_data.append(diag_vec[i] * matrix[i, j])

    return DenseTensor((rank, n), data=result_data)