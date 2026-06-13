# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    d = tt.order
    cores = []
    current = None

    for k in range(d - 1):
        core = tt.cores[k].copy()
        r_left, n_k, r_right = core.shape

        if current is not None:
            new_data = []
            for i in range(current.shape[0]):
                for j in range(n_k):
                    for p in range(r_right):
                        s = 0.0
                        for q in range(r_left):
                            s += current[i, q] * core[q, j, p]
                        new_data.append(s)
            core = DenseTensor((current.shape[0], n_k, r_right), data=new_data)
            r_left = current.shape[0]

        M = core.reshape((r_left * n_k, r_right))
        Q, R = backend.qr(M)
        new_core = Q.reshape((r_left, n_k, Q.shape[1]))
        cores.append(new_core)
        current = R

    last_core = tt.cores[-1].copy()
    if current is not None:
        r_left_last, n_last, r_right_last = last_core.shape
        new_data = []
        for i in range(current.shape[0]):
            for j in range(n_last):
                for p in range(r_right_last):
                    s = 0.0
                    for q in range(r_left_last):
                        s += current[i, q] * last_core[q, j, p]
                    new_data.append(s)
        last_core = DenseTensor((current.shape[0], n_last, r_right_last), data=new_data)
    cores.append(last_core)

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    d = tt.order
    cores = list(tt.cores)

    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        M = core.reshape((r_left, n_k * r_right))
        Mt = backend.transpose(M)
        Q, R = backend.qr(Mt)
        Q = backend.transpose(Q)
        R = backend.transpose(R)

        new_core = Q.reshape((r_left, n_k, Q.shape[1]))
        cores[k] = new_core

        prev_core = cores[k - 1]
        r_prev_left, n_prev, r_prev_right = prev_core.shape
        
        new_prev_data = []
        for i in range(r_prev_left):
            for j in range(n_prev):
                for p in range(R.shape[1]):
                    val = 0.0
                    for q in range(r_prev_right):
                        val += prev_core[i, j, q] * R[q, p]
                    new_prev_data.append(val)
        cores[k - 1] = DenseTensor((r_prev_left, n_prev, R.shape[1]), data=new_prev_data)

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.size == 0:
        return 0

    max_sigma = abs(S[0])
    threshold = max(abs_tol, rel_tol * max_sigma)

    rank = 0
    for i in range(S.size):
        if abs(S[i]) > threshold:
            rank += 1
        else:
            break

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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1:
        raise ValueError(f"_multiply_diag_matrix ожидает 1D для diag_vec, получено {diag_vec.ndim}D")
    if matrix.ndim != 2:
        raise ValueError(f"_multiply_diag_matrix ожидает 2D для matrix, получено {matrix.ndim}D")

    m, n = matrix.shape
    if m != rank or diag_vec.size != rank:
        raise ValueError("wrong sizes")

    result_data = []
    for i in range(rank):
        for j in range(n):
            result_data.append(diag_vec[i] * matrix[i, j])

    return DenseTensor((rank, n), data=result_data)


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError(f"_multiply_columns_by_diag ожидает 2D для matrix, получено {matrix.ndim}D")
    if diag_vec.ndim != 1:
        raise ValueError(f"_multiply_columns_by_diag ожидает 1D для diag_vec, получено {diag_vec.ndim}D")

    m, n = matrix.shape
    rank = diag_vec.size
    if n != rank:
        raise ValueError(f"num stolbzov {n} != rank {rank}")

    result_data = []
    for i in range(m):
        for j in range(rank):
            result_data.append(matrix[i, j] * diag_vec[j])

    return DenseTensor((m, rank), data=result_data)