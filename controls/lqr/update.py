"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def lqr_update(
        value_state_state_hessian,
        value_state_jacobian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_shift,
        cost_state_state_hessian,
        cost_state_controls_hessian,
        cost_controls_state_hessian,
        cost_controls_controls_hessian,
        cost_state_jacobian,
        cost_controls_jacobian,
):
    """Solves for the value iteration solution to lqr.

    Args:
    - value_state_state_hessian: the hessian of the cost to go wrt. state i state j
        with shape [batch_dim, state_dim, state_dim].
    - value_state_jacobian: the jacobian of the cost to go wrt. state i
        with shape [batch_dim, state_dim, 1].

    - dynamics_state_jacobian: the jacobian of the dynamics wrt. state i
        with shape [batch_dim, state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. controls i
        with shape [batch_dim, state_dim, controls_dim].

    - dynamics_shift: the shift term of the dynamics
        with shape [batch_dim, state_dim, 1].

    - cost_state_state_hessian: the hessian of the cost wrt. state i state j
        with shape [batch_dim, state_dim, state_dim].
    - cost_state_controls_hessian: the hessian of the cost wrt. state i controls j
        with shape [batch_dim, state_dim, controls_dim].
    - cost_controls_state_hessian: the hessian of the cost wrt. controls i state j
        with shape [batch_dim, controls_dim, state_dim].
    - cost_controls_controls_hessian: the hessian of the cost wrt. controls i controls j
        with shape [batch_dim, controls_dim, controls_dim].

    - cost_state_jacobian: the jacobian of the cost wrt. state i
        with shape [batch_dim, state_dim, 1].
    - cost_controls_jacobian: the jacobian of the cost wrt. controls i
        with shape [batch_dim, controls_dim, 1].

    Returns:
    - controls_state_jacobian: the jacobian of the controls with respect to the state
        with shape [batch_dim, controls_dim, state_dim].
    - controls_shift: the shift term of the controls
        with shape [batch_dim, controls_dim, 1].

    - value_state_state_hessian: the hessian of the cost to go wrt. state i state j
        with shape [batch_dim, state_dim, state_dim].
    - value_state_jacobian: the jacobian of the cost to go wrt. state i
        with shape [batch_dim, state_dim, 1].
    """

    # make sure all inputs are 3 tensors

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(value_state_state_hessian)),
        message="value_state_state_hessian should be a 3 tensor")
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(value_state_jacobian)),
        message="value_state_jacobian should be a 3 tensor")
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
        tf.size(tf.shape(dynamics_shift)),
        message="dynamics_shift should be a 3 tensor")
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_state_state_hessian)),
        message="cost_state_state_hessian should be a 3 tensor")
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_state_controls_hessian)),
        message="cost_state_controls_hessian should be a 3 tensor")
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_controls_state_hessian)),
        message="cost_controls_state_hessian should be a 3 tensor")
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_controls_controls_hessian)),
        message="cost_controls_controls_hessian should be a 3 tensor")
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_state_jacobian)),
        message="cost_state_jacobian should be a 3 tensor")
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(cost_controls_jacobian)),
        message="cost_controls_jacobian should be a 3 tensor")

    # get the batch shape and vector sizes

    controls_dim = tf.shape(dynamics_controls_jacobian)[-1]
    state_dim = tf.shape(dynamics_controls_jacobian)[-2]
    batch_dim = tf.shape(dynamics_controls_jacobian)[0]

    # make sure all inputs have the same batch shape

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(value_state_state_hessian)[0],
        message="value_state_state_hessian should have correct batch size")
    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(value_state_jacobian)[0],
        message="value_state_jacobian should have correct batch size")
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
        tf.shape(dynamics_shift)[0],
        message="dynamics_shift should have correct batch size")
    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_state_state_hessian)[0],
        message="cost_state_state_hessian should have correct batch size")
    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_state_controls_hessian)[0],
        message="cost_state_controls_hessian should have correct batch size")
    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_controls_state_hessian)[0],
        message="cost_controls_state_hessian should have correct batch size")
    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_controls_controls_hessian)[0],
        message="cost_controls_controls_hessian should have correct batch size")
    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_state_jacobian)[0],
        message="cost_state_jacobian should have correct batch size")
    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(cost_controls_jacobian)[0],
        message="cost_controls_jacobian should have correct batch size")

    # make sure all other dims are as expected

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(value_state_state_hessian)[-2],
        message="value_state_state_hessian should have shape [batch_dim, state_dim, state_dim]")
    tf.debugging.assert_equal(
        state_dim,
        tf.shape(value_state_state_hessian)[-1],
        message="value_state_state_hessian should have shape [batch_dim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(value_state_jacobian)[-2],
        message="value_state_jacobian should have shape [batch_dim, state_dim, 1]")
    tf.debugging.assert_equal(
        1,
        tf.shape(value_state_jacobian)[-1],
        message="value_state_state_hessian should have shape [batch_dim, state_dim, 1]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-2],
        message="dynamics_state_jacobian should have shape [batch_dim, state_dim, state_dim]")
    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-1],
        message="dynamics_state_jacobian should have shape [batch_dim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_controls_jacobian)[-2],
        message="dynamics_controls_jacobian should have shape [batch_dim, state_dim, controls_dim]")
    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(dynamics_controls_jacobian)[-1],
        message="dynamics_controls_jacobian should have shape [batch_dim, state_dim, controls_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_shift)[-2],
        message="dynamics_shift should have shape [batch_dim, state_dim, 1]")
    tf.debugging.assert_equal(
        1,
        tf.shape(dynamics_shift)[-1],
        message="dynamics_shift should have shape [batch_dim, state_dim, 1]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_state_hessian)[-2],
        message="cost_state_state_hessian should have shape [batch_dim, state_dim, state_dim]")
    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_state_hessian)[-1],
        message="cost_state_state_hessian should have shape [batch_dim, state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_controls_hessian)[-2],
        message="cost_state_controls_hessian should have shape [batch_dim, state_dim, controls_dim]")
    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_state_controls_hessian)[-1],
        message="cost_state_controls_hessian should have shape [batch_dim, state_dim, controls_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_state_hessian)[-2],
        message="cost_controls_state_hessian should have shape [batch_dim, controls_dim, state_dim]")
    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_controls_state_hessian)[-1],
        message="cost_controls_state_hessian should have shape [batch_dim, controls_dim, state_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_controls_hessian)[-2],
        message="cost_controls_controls_hessian should have shape [batch_dim, controls_dim, controls_dim]")
    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_controls_hessian)[-1],
        message="cost_controls_controls_hessian should have shape [batch_dim, controls_dim, controls_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_jacobian)[-2],
        message="cost_state_jacobian should have shape [batch_dim, state_dim, 1]")
    tf.debugging.assert_equal(
        1,
        tf.shape(cost_state_jacobian)[-1],
        message="cost_state_jacobian should have shape [batch_dim, state_dim, 1]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_jacobian)[-2],
        message="cost_controls_jacobian should have shape [batch_dim, controls_dim, 1]")
    tf.debugging.assert_equal(
        1,
        tf.shape(cost_controls_jacobian)[-1],
        message="cost_controls_jacobian should have shape [batch_dim, controls_dim, 1]")

    # calculate the quadratic parameters of the q function

    f_transpose_v_state = tf.matmul(
        dynamics_state_jacobian, value_state_state_hessian, transpose_a=True)
    f_transpose_v_controls = tf.matmul(
        dynamics_controls_jacobian, value_state_state_hessian, transpose_a=True)

    q_state_state_hessian = cost_state_state_hessian + tf.matmul(
        f_transpose_v_state,
        dynamics_state_jacobian)
    q_state_controls_hessian = cost_state_controls_hessian + tf.matmul(
        f_transpose_v_state,
        dynamics_controls_jacobian)
    q_controls_state_hessian = cost_controls_state_hessian + tf.matmul(
        f_transpose_v_controls,
        dynamics_state_jacobian)
    q_controls_controls_hessian = cost_controls_controls_hessian + tf.matmul(
        f_transpose_v_controls,
        dynamics_controls_jacobian)

    q_state_jacobian = cost_state_jacobian + tf.matmul(
        f_transpose_v_state,
        dynamics_shift) + tf.matmul(dynamics_state_jacobian, value_state_jacobian, transpose_a=True)
    q_controls_jacobian = cost_controls_jacobian + tf.matmul(
        f_transpose_v_controls,
        dynamics_shift) + tf.matmul(dynamics_controls_jacobian, value_state_jacobian, transpose_a=True)

    # use the q function to calculate the optimal control gain

    q_inv = tf.linalg.inv(q_controls_controls_hessian)

    controls_state_jacobian = -tf.matmul(q_inv, q_controls_state_hessian)
    controls_shift = -tf.matmul(q_inv, q_controls_jacobian)

    # calculate the quadratic parameters of the optimal value function

    k_transpose_q = tf.matmul(controls_state_jacobian, q_controls_controls_hessian, transpose_a=True)

    value_state_state_hessian = q_state_state_hessian + tf.matmul(
        q_state_controls_hessian, controls_state_jacobian) + tf.matmul(
            controls_state_jacobian, q_controls_state_hessian, transpose_a=True) + tf.matmul(
                k_transpose_q, controls_state_jacobian)

    value_state_jacobian = q_state_jacobian + tf.matmul(
        q_state_controls_hessian, controls_shift) + tf.matmul(
            controls_state_jacobian, q_controls_jacobian, transpose_a=True) + tf.matmul(
                k_transpose_q, controls_shift)

    return (
        controls_state_jacobian,
        controls_shift,
        value_state_state_hessian,
        value_state_jacobian)
