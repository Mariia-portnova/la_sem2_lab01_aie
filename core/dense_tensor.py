# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)
        
        if data is not None:
            if len(data) != self.size:
                raise ValueError(f"data length {len(data)} not fit size {self.size}")
            self.data = data.copy()
        else:
            self.data = [fill] * self.size

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        if seed is not None:
            random.seed(seed)

        size = compute_size(validate_shape(shape))
        data = []

        if integer:
            for _ in range(size):
                data.append(float(random.randint(low, high)))
        else:
            for _ in range(size):
                data.append(random.uniform(low, high))

        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        def get_shape(lst):
            shape = [len(lst)]
            if isinstance(lst[0], list):
                shape.extend(get_shape(lst[0]))
            return shape
        
        def flatten(lst, result):
            for item in lst:
                if isinstance(item, list):
                    flatten(item, result)
                else:
                    result.append(float(item))
            return result
        
        shape = get_shape(nested)
        data = []
        flatten(nested, data)
        
        return DenseTensor(tuple(shape), data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            if multi_index < 0 or multi_index >= self.size:
                raise IndexError(f"index {multi_index} out of [0, {self.size - 1}]")
            multi_index = flat_to_multi_index(multi_index, self.shape)

        if len(multi_index) != self.ndim:
            raise IndexError(
                f"num index {len(multi_index)} =! {self.ndim}"
            )

        for i, idx in enumerate(multi_index):
            if idx < 0 or idx >= self.shape[i]:
                raise IndexError(
                    f"индекс {idx} по моде {i} вне диапазона [0, {self.shape[i] - 1}]"
                )

        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        return self.data[flat_idx]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        idx_tuple = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(idx_tuple, self.strides)
        self.data[flat_idx] = value

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        new_shape_tuple = validate_shape(new_shape)
        new_size = compute_size(new_shape_tuple)

        if new_size != self.size:
            raise ValueError(
                f"new size {new_size} not fit to old one {self.size}"
            )

        return DenseTensor(new_shape_tuple, data=self.data)

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"mode {mode} out of [0, {self.ndim - 1}]")

        row_dim = self.shape[mode]
        col_dim = self.size // row_dim

        result_data = [0.0] * (row_dim * col_dim)

        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)

            row_idx = multi_idx[mode]

            other_indices = []
            for i in range(self.ndim):
                if i != mode:
                    other_indices.append(multi_idx[i])

            col_idx = 0
            stride = 1
            for i in range(len(other_indices) - 1, -1, -1):
                col_idx += other_indices[i] * stride
                stride *= self.shape[i] if i < mode else self.shape[i + 1]

            new_flat_idx = row_idx * col_dim + col_idx
            result_data[new_flat_idx] = self.data[flat_idx]

        return DenseTensor((row_dim, col_dim), data=result_data)

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"k {k} out of [0, {self.ndim - 2}]")

        row_dim = compute_size(self.shape[:k+1])
        col_dim = self.size // row_dim

        result_data = [0.0] * (row_dim * col_dim)

        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)

            row_idx = 0
            stride = 1
            for i in range(k, -1, -1):
                row_idx += multi_idx[i] * stride
                stride *= self.shape[i]

            col_idx = 0
            stride = 1
            for i in range(self.ndim - 1, k, -1):
                col_idx += multi_idx[i] * stride
                stride *= self.shape[i]

            new_flat_idx = row_idx * col_dim + col_idx
            result_data[new_flat_idx] = self.data[flat_idx]

        return DenseTensor((row_dim, col_dim), data=result_data)

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data)

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        squared_sum = 0.0
        for x in self.data:
            squared_sum += x * x
        return math.sqrt(squared_sum)

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)

        result_data = [self.data[i] + other.data[i] for i in range(self.size)]
        return DenseTensor(self.shape, data=result_data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)

        result_data = [self.data[i] - other.data[i] for i in range(self.size)]
        return DenseTensor(self.shape, data=result_data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        result_data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data=result_data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self.__mul__(-1.0)

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if self.shape != other.shape:
            return False

        for i in range(self.size):
            a = self.data[i]
            b = other.data[i]
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False

        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        def build_nested(shape, flat_data, offset):
            if len(shape) == 1:
                return flat_data[offset:offset + shape[0]]
            
            dim_size = shape[0]
            inner_shape = shape[1:]
            inner_length = compute_size(inner_shape)
            
            result = []
            for i in range(dim_size):
                inner_offset = offset + i * inner_length
                result.append(build_nested(inner_shape, flat_data, inner_offset))
            return result
        
        return build_nested(self.shape, self.data, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.data[:10]})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()