import os
import copyreg
import importlib


numpy = importlib.import_module("numpy")
xp = numpy
cupy = None


def _from_pickle(buffer: bytes, shape, dtype_str: str):
    arr = xp.frombuffer(memoryview(buffer), dtype=xp.dtype(dtype_str))
    return arr.reshape(shape).copy()


# Keep backward compatibility with old pickle files that expect numpy._from_pickle
_from_pickle.__module__ = "numpy"
setattr(numpy, "_from_pickle", _from_pickle)


def _register_cupy_pickling(cp):
    def _reduce_cupy_array(arr):
        return numpy._from_pickle, (arr.tobytes(), arr.shape, arr.dtype.str)

    copyreg.pickle(cp.ndarray, _reduce_cupy_array)


def _patch_cupy_compat(cp, np_real):
    if not hasattr(cp, "_NoValue") and hasattr(np_real, "_NoValue"):
        cp._NoValue = np_real._NoValue

    _orig_prod, _orig_size = cp.prod, cp.size

    def _prod_like(x, *args, **kwargs):
        if isinstance(x, (tuple, list)):
            p = 1
            for v in x:
                p *= int(v)
            return p
        return _orig_prod(x, *args, **kwargs)

    def _size_like(x):
        if isinstance(x, (tuple, list)):
            return _prod_like(x)
        return int(_orig_size(x))

    cp.prod = _prod_like
    cp.size = _size_like

    try:
        from cupy.lib.stride_tricks import as_strided as _cu_as_strided  # noqa: F401
        from cupy.lib.stride_tricks import sliding_window_view as _cu_slwin  # noqa: F401
        import cupy._padding.pad as _cu_pad  # noqa: F401
    except Exception:
        pass


def enable_gpu() -> None:
    global xp, cupy

    if os.getenv("SCRATCH_MODEL_CPU") == "1":
        xp = numpy
        cupy = None
        return

    try:
        import cupy as cp
    except Exception:
        xp = numpy
        cupy = None
        return

    _patch_cupy_compat(cp, numpy)
    _register_cupy_pickling(cp)

    xp = cp
    cupy = cp


def using_gpu() -> bool:
    return cupy is not None and xp is cupy


def asnumpy(arr):
    if cupy is not None and isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    return arr


def ascupy(arr):
    if cupy is not None:
        return cupy.asarray(arr)
    return numpy.asarray(arr)


enable_gpu()