# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations
import random
from typing import List, Tuple, Optional
from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not cores:
            raise ValueError("corners cannot be empty")

        self.cores = cores
        self.order = len(cores)

        ranks_list: list[int] = []
        shape_list: list[int] = []

        for k, core in enumerate(cores):
            if core.ndim != 3:
                raise ValueError(f"corner {k} must be 3D, got {core.ndim}D")

            r_left, n_k, r_right = core.shape

            shape_list.append(n_k)

            if k == 0:
                if r_left != 1:
                    raise ValueError(f"1rst corner must has r_0=1, got {r_left}")
                ranks_list.append(1)
                ranks_list.append(r_right)
            else:
                if r_left != ranks_list[-1]:
                    raise ValueError(
                        f"wrong ranks: r_{k}={r_left}, expected {ranks_list[-1]}"
                    )
                ranks_list.append(r_right)

            if k == self.order - 1 and r_right != 1:
                raise ValueError(f"last corner must has r_d=1, got {r_right}")

        self.shape = tuple(shape_list)
        self.ranks = tuple(ranks_list)


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        if seed is not None:
            random.seed(seed)

        shape = validate_shape(shape)
        d = len(shape)

        if isinstance(ranks, (tuple, list)):
            if len(ranks) == d + 1:
                if ranks[0] != 1 or ranks[-1] != 1:
                    raise ValueError("1rst and last ranks must be 1")
                full_ranks = list(ranks)
            elif len(ranks) == d - 1:
                full_ranks = [1] + list(ranks) + [1]
            else:
                raise ValueError(
                    f"wrond lenght ranks: {len(ranks)}, expected {d+1} or {d-1}"
                )
        else:
            raise TypeError("ranks must be tuple or list")

        cores = []
        for k in range(d):
            r_left = full_ranks[k]
            n_k = shape[k]
            r_right = full_ranks[k + 1]

            core = DenseTensor.random((r_left, n_k, r_right), seed=seed)
            cores.append(core)

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if len(indices) != self.order:
            raise ValueError(
                f"num index {len(indices)} != order {self.order}"
            )

        result: float = 1.0
        current_matrix = None

        for k, idx in enumerate(indices):
            core = self.cores[k]

            if idx < 0 or idx >= core.shape[1]:
                raise IndexError(
                    f"index {idx} mod {k} out of [0, {core.shape[1] - 1}]"
                )

            slice_matrix = DenseTensor.zeros((core.shape[0], core.shape[2]))

            for i in range(core.shape[0]):
                for j in range(core.shape[2]):
                    slice_matrix[i, j] = core[i, idx, j]

            if current_matrix is None:
                current_matrix = slice_matrix
            else:
                new_matrix = DenseTensor.zeros((current_matrix.shape[0], slice_matrix.shape[1]))
                for i in range(current_matrix.shape[0]):
                    for j in range(slice_matrix.shape[1]):
                        s: float = 0.0
                        for p in range(current_matrix.shape[1]):
                            s += current_matrix[i, p] * slice_matrix[p, j]
                        new_matrix[i, j] = s
                current_matrix = new_matrix

        if current_matrix is not None and current_matrix.shape == (1, 1):
            result = current_matrix[0, 0]

        return result

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = None

        for k in range(self.order):
            core = self.cores[k]
            n_k = core.shape[1]

            slices = []
            for i in range(n_k):
                slice_mat = DenseTensor.zeros((core.shape[0], core.shape[2]))
                for p in range(core.shape[0]):
                    for q in range(core.shape[2]):
                        slice_mat[p, q] = core[p, i, q]
                slices.append(slice_mat)

            if result is None:
                result = slices
            else:
                new_result = []
                for prev in result:
                    for curr in slices:
                        prod = DenseTensor.zeros((prev.shape[0], curr.shape[1]))
                        for i in range(prev.shape[0]):
                            for j in range(curr.shape[1]):
                                s: float = 0.0
                                for p in range(prev.shape[1]):
                                    s += prev[i, p] * curr[p, j]
                                prod[i, j] = s
                        new_result.append(prod)
                result = new_result

        if result is None:
            return DenseTensor.zeros((0,))

        full_shape = self.shape
        full_size = compute_size(full_shape)
        full_data = [0.0] * full_size

        from core.utils import flat_to_multi_index

        for flat_idx in range(full_size):
            multi_idx = flat_to_multi_index(flat_idx, full_shape)
            value = self.get_element(multi_idx)
            full_data[flat_idx] = value

        return DenseTensor(full_shape, data=full_data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        total: int = 0
        for core in self.cores:
            total += core.size
        return total

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        dense_size = compute_size(self.shape)
        tt_size = self.total_storage()
        if tt_size == 0:
            return float('inf')
        return dense_size / tt_size

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        copied_cores = [core.copy() for core in self.cores]
        return TTTensor(copied_cores)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        lines = []
        lines.append(f"TTTensor(order={self.order}, shape={self.shape}, ranks={self.ranks})")
        lines.append(f"  core shapes: {self.core_sizes()}")
        lines.append(f"  total storage: {self.total_storage()} elements")
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()