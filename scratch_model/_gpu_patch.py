import os
import sys
import importlib
import copyreg


def _from_pickle(buffer: bytes, shape, dtype_str: str):
    """Rebuild array using whichever 'numpy' is currently active (CuPy or NumPy)."""
    xp = sys.modules.get("numpy")
    arr = xp.frombuffer(memoryview(buffer), dtype=xp.dtype(dtype_str))
    return arr.reshape(shape).copy()


_from_pickle.__module__ = "numpy"


def _register_cupy_pickling(cp):
    # publish loader on the active numpy module
    setattr(sys.modules["numpy"], "_from_pickle", _from_pickle)

    def _reduce_cupy_array(arr):
        # refer to the *importable* numpy._from_pickle, not a local function
        return sys.modules["numpy"]._from_pickle, (arr.tobytes(), arr.shape, arr.dtype.str)

    copyreg.pickle(cp.ndarray, _reduce_cupy_array)


def enable_gpu() -> None:
    if os.getenv("SCRATCH_MODEL_CPU") == "1":
        return

    np_real = importlib.import_module("numpy")

    try:
        import cupy as cp
    except Exception:
        return

    try:
        from cupy.lib.stride_tricks import as_strided as _cu_as_strided
        from cupy.lib.stride_tricks import sliding_window_view as _cu_slwin
        import cupy._padding.pad as _cu_pad
    except Exception:
        pass

    if not hasattr(cp, "_NoValue") and hasattr(np_real, "_NoValue"):
        cp._NoValue = np_real._NoValue  # type: ignore[attr-defined]

    sys.modules["numpy"] = cp
    # sys.modules["numpy_original"] = np_real
    _register_cupy_pickling(cp)

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


enable_gpu()
