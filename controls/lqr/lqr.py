"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.update import update
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
        with shape [T, batch_dim, state_dim].
    - Cu: the jacobian of the cost wrt. controls i
        with shape [T, batch_dim, controls_dim].

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
        with shape [T, batch_dim, state_dim].
    - Qu: the jacobian of the cost to go wrt. controls i
        with shape [T, batch_dim, controls_dim].

    - K: the jacobian of the controls with respect to the state
        with shape [T, batch_dim, controls_dim, state_dim].
    - k: the shift term of the controls
        with shape [T, batch_dim, controls_dim].
    - S: covariance of the maximum entropy controls
        with shape [T, batch_dim, controls_dim, controls_dim].

    - Vxx: the hessian of the cost to go wrt. state i state j
        with shape [T, batch_dim, state_dim, state_dim].
    - Vx: the jacobian of the cost to go wrt. state i
        with shape [T, batch_dim, state_dim].
    """

    def body(
            *inputs
    ):
        Qxx, Qxu, Qux, Quu, Qx, Qu, Kx, k, S, Vxx, Vx = update(
            inputs[0],
            inputs[1],
            inputs[2][inputs[21], :, :, :],
            inputs[3][inputs[21], :, :, :],
            inputs[4][inputs[21], :, :, :],
            inputs[5][inputs[21], :, :, :],
            inputs[6][inputs[21], :, :, :],
            inputs[7][inputs[21], :, :, :],
            inputs[8][inputs[21], :, :, tf.newaxis],
            inputs[9][inputs[21], :, :, tf.newaxis])

        return (
            Vxx,
            Vx,
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
            inputs[9],
            inputs[10].write(inputs[21], Qxx),
            inputs[11].write(inputs[21], Qxu),
            inputs[12].write(inputs[21], Qux),
            inputs[13].write(inputs[21], Quu),
            inputs[14].write(inputs[21], Qx[:, :, 0]),
            inputs[15].write(inputs[21], Qu[:, :, 0]),
            inputs[16].write(inputs[21], Kx),
            inputs[17].write(inputs[21], k[:, :, 0]),
            inputs[18].write(inputs[21], S),
            inputs[19].write(inputs[21], Vxx),
            inputs[20].write(inputs[21], Vx[:, :, 0]),
            inputs[21] - 1,
            inputs[22])

    lqr_results = tf.while_loop(
        lambda *inputs: tf.greater_equal(inputs[21], 0),
        body, (
            tf.zeros_like(Cxx[0, :, :, :]),
            tf.zeros_like(Cx[0, :, :])[:, :, tf.newaxis],
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
        lqr_results[19].stack(),
        lqr_results[20].stack())
