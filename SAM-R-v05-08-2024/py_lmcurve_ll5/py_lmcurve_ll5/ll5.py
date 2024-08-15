import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import math
from dataclasses import dataclass
from typing import Optional
import os 


library_path = os.path.join(os.path.dirname(__file__), 'r_lmcurve_ll5.so')

# Load the library
lib = ctypes.CDLL(library_path)

# Define the function types
class Context(ctypes.Structure):
    _fields_ = [
        ("b", ctypes.c_double),
        ("c", ctypes.c_double),
        ("d", ctypes.c_double),
        ("e", ctypes.c_double),
        ("f", ctypes.c_double),
    ]


@dataclass
class ll5Params:
    b: Optional[float]
    c: Optional[float]
    d: Optional[float]
    e: Optional[float]
    f: Optional[float]


# Function signature
lib.r_lmcurve_ll5.argtypes = [
    ndpointer(ctypes.c_double),  # x array
    ndpointer(ctypes.c_double),  # y array
    ctypes.POINTER(ctypes.c_int),  # n, length of x and y
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # c
    ctypes.POINTER(ctypes.c_double),  # d
    ctypes.POINTER(ctypes.c_double),  # e
    ctypes.POINTER(ctypes.c_double),  # f
]
lib.r_lmcurve_ll5.restype = None


def lmcurve_ll5(x, y, b=None, c=None, d=None, e=None, f=None) -> ll5Params:
    n = ctypes.c_int(len(x))
    x_array = np.array(x, dtype=np.double)
    y_array = np.array(y, dtype=np.double)

    # Convert None to NaN for C compatibility
    b = ctypes.c_double(float("nan") if b is None else b)
    c = ctypes.c_double(float("nan") if c is None else c)
    d = ctypes.c_double(float("nan") if d is None else d)
    e = ctypes.c_double(float("nan") if e is None else e)
    f = ctypes.c_double(float("nan") if f is None else f)

    # Call the C function
    lib.r_lmcurve_ll5(
        x_array,
        y_array,
        ctypes.byref(n),
        ctypes.byref(b),
        ctypes.byref(c),
        ctypes.byref(d),
        ctypes.byref(e),
        ctypes.byref(f),
    )

    # Convert back to Python types and check for NaN to return None

    return ll5Params(
        None if math.isnan(b.value) else b.value,
        None if math.isnan(c.value) else c.value,
        None if math.isnan(d.value) else d.value,
        None if math.isnan(e.value) else e.value,
        None if math.isnan(f.value) else f.value,
    )


if __name__ == "__main__":
    a = lmcurve_ll5([1, 2, 3], [1, 2, 3], b=1, c=2, d=3, e=4, f=5)
    print(a)