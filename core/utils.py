# core/utils.py

"""Вспомогательные функции для работы с тензорами."""


def validate_shape(
    shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.

    Убеждается, что shape является последовательностью положительных целых
    чисел. Преобразует список в кортеж для единообразия.

    Args:
        shape: кортеж или список размеров тензора по каждой моде

    Returns:
        tuple: проверенный кортеж положительных целых чисел

    Raises:
        TypeError:  если shape не является tuple или list
        ValueError: если хотя бы один элемент shape не является
                    положительным целым числом
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"shape must be tuple or list, got {type(shape)}")

    if len(shape) == 0:
        raise ValueError("shape cannot be empty")

    for dim in shape:
        if not isinstance(dim, int):
            raise ValueError(f"wrong size, got {type(dim)}")
        if dim <= 0:
            raise ValueError(f"size must be >0, got {dim}")

    return tuple(shape)



def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    size: int = 1
    for dim in shape:
        size *= dim
    return size


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает кортеж strides, содержащий для каждой моды k свой strides[k].

    Stride по моде k — это число элементов в плоском списке, на которое
    нужно сдвинуться, чтобы перейти к следующему элементу вдоль моды k.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    ndim: int = len(shape)
    strides: list[int] = [0] * ndim

    stride: int = 1
    for k in range(ndim - 1, -1, -1):
        strides[k] = stride
        stride *= shape[k]

    return tuple(strides)


def multi_index_to_flat(
    multi_index: tuple[int, ...],
    strides: tuple[int, ...]
) -> int:
    """
    Возвращает позицию элемента в плоском списке данных по его
    многомерным координатам и заранее вычисленным strides.

    Args:
        multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1})
        strides:     кортеж шагов   (s_0, s_1, ..., s_{d-1})
    """
    if len(multi_index) != len(strides):
        raise ValueError("num index != num strides")

    flat_index: int = 0
    for idx, stride in zip(multi_index, strides):
        flat_index += idx * stride

    return flat_index


def flat_to_multi_index(
    flat_index: int,
    shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Возвращает мультииндекс на основе плоского индекса.

    Args:
        flat_index: плоский индекс в списке данных
        shape:      кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    if flat_index < 0:
        raise ValueError("flat_index не может быть отрицательным")

    size: int = compute_size(shape)
    if flat_index >= size:
        raise ValueError(f"flat_index {flat_index} out of [0, {size - 1}]")

    multi_index: list[int] = [0] * len(shape)
    remainder: int = flat_index

    for k in range(len(shape) - 1, -1, -1):
        multi_index[k] = remainder % shape[k]
        remainder //= shape[k]

    return tuple(multi_index)

def check_shapes_match(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...]
) -> None:
    """
    Проверяет совпадение форм двух тензоров.

    Используется перед поэлементными операциями (сложение, вычитание),
    чтобы гарантировать совместимость тензоров.

    Args:
        shape1: кортеж размеров первого тензора
        shape2: кортеж размеров второго тензора

    Raises:
        ValueError: если формы не совпадают
    """
    if shape1 != shape2:
        raise ValueError(f"not same forms: {shape1} != {shape2}")
