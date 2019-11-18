"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.lqr import lqr
import tensorflow as tf


def infinite_horizon_lqr(
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
        max_iterations=100
):
    """Solves for the value iteration solution to lqr.

    Args:
    - dynamics_state_jacobian: the jacobian of the dynamics wrt. state i
        with shape [batch_dim, ..., state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. controls i
        with shape [batch_dim, ..., state_dim, controls_dim].

    - cost_state_hessian: the hessian of the cost wrt. state i state j
        with shape [batch_dim, ..., state_dim, state_dim].
    - cost_controls_hessian: the hessian of the cost wrt. controls i controls j
        with shape [batch_dim, ..., controls_dim, controls_dim].

    Returns:
    - controls_jacobian: the jacobian of the controls with respect to the state
        with shape [batch_dim, ..., controls_dim, state_dim].

    - value_state_hessian: the hessian of the cost to go wrt. state i state j
        with shape [batch_dim, ..., state_dim, state_dim].
        """

    batch_shape = tf.shape(dynamics_state_jacobian)[:-2]

    state_dim = tf.shape(dynamics_state_jacobian)[-1]

    controls_dim = tf.shape(dynamics_controls_jacobian)[-1]

    dtype = dynamics_controls_jacobian.dtype

    dynamics_state_jacobian = tf.tile(
        dynamics_state_jacobian[tf.newaxis, :, :, :], [max_iterations, 1, 1, 1])

    dynamics_controls_jacobian = tf.tile(
        dynamics_controls_jacobian[tf.newaxis, :, :, :], [max_iterations, 1, 1, 1])

    cost_state_hessian = tf.tile(
        cost_state_hessian[tf.newaxis, :, :, :], [max_iterations, 1, 1, 1])

    cost_controls_hessian = tf.tile(
        cost_controls_hessian[tf.newaxis, :, :, :], [max_iterations, 1, 1, 1])

    (controls_state_jacobian,
        controls_shift,
        value_state_state_hessian,
        value_state_jacobian) = lqr(
            dynamics_state_jacobian,
            dynamics_controls_jacobian,
            tf.zeros(tf.concat([[max_iterations], batch_shape, [state_dim, 1]], 0), dtype=dtype),
            cost_state_hessian,
            tf.zeros(tf.concat([[max_iterations], batch_shape, [state_dim, controls_dim]], 0), dtype=dtype),
            tf.zeros(tf.concat([[max_iterations], batch_shape, [controls_dim, state_dim]], 0), dtype=dtype),
            cost_controls_hessian,
            tf.zeros(tf.concat([[max_iterations], batch_shape, [state_dim, 1]], 0), dtype=dtype),
            tf.zeros(tf.concat([[max_iterations], batch_shape, [controls_dim, 1]], 0), dtype=dtype))

    return (
        controls_state_jacobian[0],
        value_state_state_hessian[0])
