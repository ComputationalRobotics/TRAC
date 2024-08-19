import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import NamedTuple, Optional
import chex
from optax import tree_utils
from optax._src import base
from optax._src import utils

# Neither JAX or PyTorch have built in Erfi support.
# Therefore, we convert https://github.com/redsnic/torch_erf to Jax.

class ERF_1994:
    """Class to compute the error function of a complex number (extends jax.special.erf behavior)

    This class is based on the algorithm proposed in:
    Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518
    """
    def __init__(self, n_coefs):
        self.N, self.M = n_coefs, 2 * n_coefs
        self.L = jnp.sqrt(self.N / jnp.sqrt(2.))
        k = jnp.arange(-self.M + 1, self.M)
        t = self.L * jnp.tan(k * jnp.pi / (2 * self.M))
        f = jnp.exp(-t**2) * (self.L**2 + t**2)
        self.a = jnp.fft.fft(jnp.fft.fftshift(f)).real[:self.N] / (4 * self.M)
        self.a = self.a[:0:-1]

    @partial(jit, static_argnums=(0,))
    def w_algorithm(self, z):
        Z = (self.L + 1j * z) / (self.L - 1j * z)
        p = jnp.polyval(self.a, Z)
        return 2 * p / (self.L - 1j * z)**2 + (1 / jnp.sqrt(jnp.pi)) / (self.L - 1j * z)

    @partial(jit, static_argnums=(0,))
    def forward(self, z):
        sign_r, sign_i = jnp.sign(z.real), jnp.sign(z.imag)
        z = jnp.abs(z.real) + 1j * jnp.abs(z.imag)
        out = -jnp.exp(jnp.log(self.w_algorithm(z * 1j)) - z**2) + 1
        return out.real * sign_r + 1j * out.imag * sign_i

    @partial(jit, static_argnums=(0,))
    def backward(self, z):
        return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-z**2)

erf_jax = ERF_1994(128)
@jit
def erfi(x):
    x = x.astype(jnp.float32) if not jnp.issubdtype(x.dtype, jnp.floating) else x
    return erf_jax.forward(1j * x).imag
csi = erfi(1.0 / jnp.sqrt(2.0))
sqrt2 = jnp.sqrt(2.0)

class TracState(NamedTuple):
  base_optimizer_state: base.OptState
  count: chex.Array
  sigma: chex.Array
  variance: chex.Array
  s: chex.Array
  theta_ref: base.Updates
  betas: chex.Array

def start_trac(optimizer: base.GradientTransformation, eps: float = 1e-8, s_prev: float = 1e-8, num_betas: int = 6) -> base.GradientTransformation:
  """
  
  TRAC - a Parameter-free optimizer which uses the erfi-tuner from Zhang et. al 2024.
  Website: https://computationalrobotics.seas.harvard.edu/TRAC/
  Paper: https://arxiv.org/abs/2405.16642
  Code: https://github.com/ComputationalRobotics/TRAC?tab=readme-ov-file
  We closely follow the meta-optimizer structure from the code in Cutkosky et. al 2023.
  
  Example Usage: optimizer = start_trac(optimizer=optax.adam(1e-3))
  
  """

  def init_fn(params: base.Params) -> TracState:
      return TracState(
          base_optimizer_state=optimizer.init(params),
          count=jnp.zeros([], jnp.int32),
          sigma=jnp.ones([num_betas,], jnp.float32) * 1e-8,
          variance=jnp.zeros([num_betas,], jnp.float32),
          s=jnp.ones([num_betas,], jnp.float32) * s_prev,
          theta_ref=params,
          betas=jnp.array([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
      )

  def update_fn(updates: base.Updates, state: TracState, params: Optional[base.Params] = None) -> tuple[base.Params, TracState]:
      if params is None:
          raise ValueError(base.NO_PARAMS_MSG)
      
      count_inc = utils.safe_int32_increment(state.count)
      new_neg_updates, base_optimizer_state = optimizer.update(updates, state.base_optimizer_state, params)
      
      theta_ref = jax.lax.cond(state.count == 0, lambda: params, lambda: state.theta_ref)
      s_sum = jnp.sum(state.s)
      
      delta_prev = jax.tree_util.tree_map(lambda xti, x0i: (x0i - xti) / (s_sum + eps), params, theta_ref)
      delta = jax.tree_util.tree_map(lambda si, ui: si - ui, delta_prev, new_neg_updates)
      
      h = tree_utils.tree_vdot(updates, delta_prev)
      variance = (state.betas**2) * state.variance + h**2
      sigma = (state.betas * state.sigma) - (-h)
      
      s = (s_prev / csi) * erfi(sigma / ((sqrt2 * jnp.sqrt(variance)) + eps))
      s_sum = jnp.sum(s)
      new_params = jax.tree_util.tree_map(lambda x, d: x - s_sum * d, theta_ref, delta)
      new_neg_updates = jax.tree_util.tree_map(lambda np, p: np - p, new_params, params)
      
      return new_neg_updates, TracState(base_optimizer_state, count_inc, sigma, variance, s, theta_ref, state.betas)

  return base.GradientTransformation(init_fn, update_fn)