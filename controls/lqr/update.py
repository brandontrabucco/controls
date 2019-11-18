"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def lqr_update(
        value_hessian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
):
    """Solves for the value iteration solution to lqr.

    Args:
    - value_hessian: the hessian of the cost to go wrt. the state
        with shape [batch_dim, state_dim, state_dim].

    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the state
        with shape [batch_dim, state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls
        with shape [batch_dim, state_dim, controls_dim].

    - cost_state_hessian: the hessian of the cost wrt. the state
        with shape [batch_dim, state_dim, state_dim].
    - cost_controls_hessian: the hessian of the cost wrt. the controls
        with shape [batch_dim, controls_dim, controls_dim].

    Returns:
    - controls_state_jacobian: the jacobian of the controls with respect to the state
        with shape [batch_dim, controls_dim, state_dim].
    - value_hessian: the hessian of the cost to go wrt. the state
        with shape [batch_dim, state_dim, state_dim].
        """

    # get the batch shape and vector sizes

    controls_dim = tf.shape(dynamics_controls_jacobian)[-1]

    state_dim = tf.shape(dynamics_state_jacobian)[-1]

    batch_dim = tf.shape(value_hessian)[0]

    # make sure all inputs are 3 tensors

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(value_hessian)),
        message="value_hessian should be a 3 tensor")

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(dynamics_state_jacobian)),
        message="dynamics_state_jacobian should be a 3 tensor")

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(dynamics_controls_jacobian)),
        message="dynamics_controls_jacobian should be a 3 tensor")

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_state_hessian)),
        message="cost_state_hessian should be a 3 tensor")

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_controls_hessian)),
        message="cost_controls_hessian should be a 3 tensor")

    # make sure all inputs have the same batch shape

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(value_hessian)[0],
        message="value_hessian should have correct batch size")

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(dynamics_state_jacobian)[0],
        message="dynamics_state_jacobian should have correct batch size")

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(dynamics_controls_jacobian)[0],
        message="dynamics_controls_jacobian should have correct batch size")

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_state_hessian)[0],
        message="cost_state_hessian should have correct batch size")

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_controls_hessian)[0],
        message="cost_controls_hessian should have correct batch size")

    # make sure all other dims are as expected

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(value_hessian)[-2],
        message="value_hessian should have shape [batch_sim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(value_hessian)[-1],
        message="value_hessian should have shape [batch_sim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-2],
        message="dynamics_state_jacobian should have shape [batch_sim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-1],
        message="dynamics_state_jacobian should have shape [batch_sim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_controls_jacobian)[-2],
        message="dynamics_controls_jacobian should have shape [batch_sim, state_dim, controls_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(dynamics_controls_jacobian)[-1],
        message="dynamics_controls_jacobian should have shape [batch_sim, state_dim, controls_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_hessian)[-2],
        message="cost_state_hessian should have shape [batch_sim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_hessian)[-1],
        message="cost_state_hessian should have shape [batch_sim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_hessian)[-2],
        message="cost_controls_hessian should have shape [batch_sim, controls_dim, controls_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_hessian)[-1],
        message="cost_controls_hessian should have shape [batch_sim, controls_dim, controls_dim]")

    # calculate the optimal control gain for the current state

    b_transpose_p = tf.matmul(
        dynamics_controls_jacobian,
        value_hessian, transpose_a=True)

    controls_state_jacobian = -tf.matmul(
        tf.linalg.inv(cost_controls_hessian + tf.matmul(
            b_transpose_p, dynamics_controls_jacobian)), tf.matmul(
                b_transpose_p, dynamics_state_jacobian))

    # calculate the optimal cost to go hessian at the current state

    a_plus_bk = dynamics_state_jacobian + tf.matmul(
        dynamics_controls_jacobian,
        controls_state_jacobian)

    value_hessian = cost_state_hessian + tf.matmul(
        tf.matmul(controls_state_jacobian, cost_controls_hessian, transpose_a=True),
        controls_state_jacobian) + tf.matmul(
            tf.matmul(a_plus_bk, value_hessian, transpose_a=True), a_plus_bk)

    return (
        controls_state_jacobian,
        value_hessian)
