"""Numerical gates must reject NaN as well as finite out-of-tolerance values."""
import math

INT8_REFERENCE_TOLERANCE = 2e-3

def int8_reference_pass(errors):
    return bool(errors) and all(math.isfinite(v) and v < INT8_REFERENCE_TOLERANCE for v in errors)
