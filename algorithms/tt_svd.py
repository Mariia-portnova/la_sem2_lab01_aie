# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math
from core.utils import flat_to_multi_index, compute_size
    
from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    d = tensor.ndim
    tensor_norm = tensor.norm()

    if tensor_norm < 1e-30:
        delta = 0.0
    else:
        delta = eps * tensor_norm / math.sqrt(d - 1) if d > 1 else 0.0

    cores = []
    current = tensor.copy()
    r_left = 1

    for k in range(d - 1):
        unfolding = current.left_unfolding(k)
        
        U, S, Vt = backend.svd(unfolding)

        rank = _compute_truncated_rank(S, delta, max_rank)

        if rank == 0:
            rank = 1

        U_trunc = _truncate_columns(U, rank, backend)

        core_shape = (r_left, tensor.shape[k], rank)
        core = DenseTensor.zeros(core_shape)
        for i in range(r_left):
            for j in range(tensor.shape[k]):
                for p in range(rank):
                    core[i, j, p] = U_trunc[i * tensor.shape[k] + j, p]

        cores.append(core)

        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        current = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)
        
        r_left = rank

    last_shape = (r_left, tensor.shape[d - 1], 1)
    last_core = DenseTensor.zeros(last_shape)
    
    for i in range(r_left):
        for j in range(tensor.shape[d - 1]):
            last_core[i, j, 0] = current[i, j]

    cores.append(last_core)

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
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

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError(f"_truncate_columns exp 2D, got {matrix.ndim}D")

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
        raise ValueError(f"_truncate_rows exp 2D, got {matrix.ndim}D")

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
        raise ValueError(f"_truncate_vector exp 1D, got {vector.ndim}D")

    if rank < 0 or rank > vector.size:
        raise ValueError(f"rank {rank} out of [0, {vector.size}]")

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
        raise ValueError(f"_multiply_diag_matrix exp 1D для diag_vec, got {diag_vec.ndim}D")
    if matrix.ndim != 2:
        raise ValueError(f"_multiply_diag_matrix exp 2D для matrix, got {matrix.ndim}D")

    m, n = matrix.shape
    if m != rank or diag_vec.size != rank:
        raise ValueError("wrong sizes")

    result_data = []
    for i in range(rank):
        for j in range(n):
            result_data.append(diag_vec[i] * matrix[i, j])

    return DenseTensor((rank, n), data=result_data)