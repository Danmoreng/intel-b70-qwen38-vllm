"""Narrow B70-specific operators with an explicit fallback contract."""
from .q128_attention import flash_attn_varlen_func, load_library
__all__ = ["flash_attn_varlen_func", "load_library"]
