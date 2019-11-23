"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.body import lqr_body
from controls.lqr.condition import lqr_condition
import tensorflow as tf


def lqr(
        Fx,
        Fu,
        Cxx,
        Cxu,
        Cux,
        Cuu,
        Cx,
        Cu,
):
    """Solves for the value iteration solution to lqr.

    Args:
    - Fx: the jacobian of the dynamics wrt. state i
        with shape [T, batch_dim, state_dim, state_dim].
    - Fu: the jacobian of the dynamics wrt. controls i
        with shape [T, batch_dim, state_dim, controls_dim].

    - Cxx: the hessian of the cost wrt. state i state j
        with shape [T, batch_dim, state_dim, state_dim].
    - Cxu: the hessian of the cost wrt. state i controls j
        with shape [T, batch_dim, state_dim, controls_dim].
    - Cux: the hessian of the cost wrt. controls i state j
        with shape [T, batch_dim, controls_dim, state_dim].
    - Cuu: the hessian of the cost wrt. controls i controls j
        with shape [T, batch_dim, controls_dim, controls_dim].

    - Cx: the jacobian of the cost wrt. state i
        with shape [T, batch_dim, state_dim, 1].
    - Cu: the jacobian of the cost wrt. controls i
        with shape [T, batch_dim, controls_dim, 1].

    Returns:
    - Qxx: the hessian of the cost to go wrt. state i state j
        with shape [T, batch_dim, state_dim, state_dim].
    - Qxu: the hessian of the cost to go wrt. state i controls j
        with shape [T, batch_dim, state_dim, controls_dim].
    - Qux: the hessian of the cost to go wrt. controls i state j
        with shape [T, batch_dim, controls_dim, state_dim].
    - Quu: the hessian of the cost to go wrt. controls i controls j
        with shape [T, batch_dim, controls_dim, controls_dim].

    - Qx: the jacobian of the cost to go wrt. state i
        with shape [T, batch_dim, state_dim, 1].
    - Qu: the jacobian of the cost to go wrt. controls i
        with shape [T, batch_dim, controls_dim, 1].

    - K: the jacobian of the controls with respect to the state
        with shape [T, batch_dim, controls_dim, state_dim].
    - k: the shift term of the controls
        with shape [T, batch_dim, controls_dim, 1].

    - Vxx: the hessian of the cost to go wrt. state i state j
        with shape [T, batch_dim, state_dim, state_dim].
    - Vx: the jacobian of the cost to go wrt. state i
        with shape [T, batch_dim, state_dim, 1].
    """
    lqr_results = tf.while_loop(
        lqr_condition,
        lqr_body, (
            tf.zeros_like(Cxx[0, :, :, :]),
            tf.zeros_like(Cx[0, :, :, :]),
            Fx,
            Fu,
            Cxx,
            Cxu,
            Cux,
            Cuu,
            Cx,
            Cu,
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.TensorArray(Fx.dtype, size=tf.shape(Fx)[0]),
            tf.shape(Fx)[0] - 1,
            tf.shape(Fx)[0]))

    return (
        lqr_results[10].stack(),
        lqr_results[11].stack(),
        lqr_results[12].stack(),
        lqr_results[13].stack(),
        lqr_results[14].stack(),
        lqr_results[15].stack(),
        lqr_results[16].stack(),
        lqr_results[17].stack(),
        lqr_results[18].stack(),
        lqr_results[19].stack())
