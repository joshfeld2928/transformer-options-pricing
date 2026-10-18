"""
One-time JAX configuration for CPU training.

- Persistent compilation cache in ``<repo>/.jax_cache`` so repeat runs skip
  most XLA compile time.
- float32 everywhere (``jax_enable_x64`` stays off).
- The ``rbg`` PRNG, which makes dropout masks roughly 2x cheaper on CPU
  than the default threefry (whole training steps run ~25% faster).

``XLA_FLAGS`` is left to the user. For these small models
``XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"``
is sometimes faster; benchmark before adopting it.
"""

from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".jax_cache"

_configured = False


def configure_jax():
    global _configured
    if _configured:
        return
    CACHE_DIR.mkdir(exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_prng_impl", "rbg")
    _configured = True
