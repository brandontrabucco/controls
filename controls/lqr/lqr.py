"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.body import lqr_body
from controls.lqr.condition import lqr_condition
import tensorflow as tf


def lqr(
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian
):
    """Solves for the value iteration solution to lqr.

    Args:
    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the state
        with shape [T, batch_dim, ..., state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls
        with shape [T, batch_dim, ..., state_dim, controls_dim].

    - cost_state_hessian: the hessian of the cost wrt. the state
        with shape [T, batch_dim, ..., state_dim, state_dim].
    - cost_controls_hessian: the hessian of the cost wrt. the controls
        with shape [T, batch_dim, ..., controls_dim, controls_dim].

    Returns:
    - controls_state_jacobians: the jacobians of the controls with respect to the states
        with shape [T, batch_dim, ..., controls_dim, state_dim].
    - value_hessians: the hessians of the cost to go wrt. the states
        with shape [T, batch_dim, ..., state_dim, state_dim].
        """

    # get the batch shape and vector sizes

    horizon = tf.shape(dynamics_state_jacobian)[0]

    batch_shape = tf.shape(dynamics_state_jacobian)[1:-2]

    batch_dim = tf.reduce_prod(batch_shape)

    state_dim = tf.shape(dynamics_state_jacobian)[-1]

    controls_dim = tf.shape(dynamics_controls_jacobian)[-1]

    dtype = dynamics_controls_jacobian.dtype

    # check the horizon of every tensor

    tf.debugging.assert_equal(
        horizon,
        tf.shape(dynamics_state_jacobian)[0],
        message="dynamics_state_jacobian should have correct horizon")

    tf.debugging.assert_equal(
        horizon,
        tf.shape(dynamics_controls_jacobian)[0],
        message="dynamics_controls_jacobian should have correct horizon")

    tf.debugging.assert_equal(
        horizon,
        tf.shape(cost_state_hessian)[0],
        message="cost_state_hessian should have correct horizon")

    tf.debugging.assert_equal(
        horizon,
        tf.shape(cost_controls_hessian)[0],
        message="cost_controls_hessian should have correct horizon")

    # make sure all inputs have the same batch shape

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(dynamics_state_jacobian)[1:-2],
        message="dynamics_state_jacobian should have correct batch shape")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(dynamics_controls_jacobian)[1:-2],
        message="dynamics_controls_jacobian should have correct batch shape")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(cost_state_hessian)[1:-2],
        message="cost_state_hessian should have correct batch shape")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(cost_controls_hessian)[1:-2],
        message="cost_controls_hessian should have correct batch shape")

    # make sure all other dims are as expected

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-2],
        message="dynamics_state_jacobian should have shape [T, batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-1],
        message="dynamics_state_jacobian should have shape [T, batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_controls_jacobian)[-2],
        message="dynamics_controls_jacobian should have shape [T, batch_sim, ..., state_dim, controls_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(dynamics_controls_jacobian)[-1],
        message="dynamics_controls_jacobian should have shape [T, batch_sim, ..., state_dim, controls_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_hessian)[-2],
        message="cost_state_hessian should have shape [T, batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_hessian)[-1],
        message="cost_state_hessian should have shape [T, batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_hessian)[-2],
        message="cost_controls_hessian should have shape [T, batch_sim, ..., controls_dim, controls_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_hessian)[-1],
        message="cost_controls_hessian should have shape [T, batch_sim, ..., controls_dim, controls_dim]")

    # flatten the batch shape

    dynamics_state_jacobian = tf.reshape(
        dynamics_state_jacobian, [horizon, batch_dim, state_dim, state_dim])

    dynamics_controls_jacobian = tf.reshape(
        dynamics_controls_jacobian, [horizon, batch_dim, state_dim, controls_dim])

    cost_state_hessian = tf.reshape(
        cost_state_hessian, [horizon, batch_dim, state_dim, state_dim])

    cost_controls_hessian = tf.reshape(
        cost_controls_hessian, [horizon, batch_dim, controls_dim, controls_dim])

    # create the loop variables

    initial_controls_state_jacobian = tf.zeros(
        [batch_dim, controls_dim, state_dim], dtype=dtype)

    initial_value_hessian = tf.zeros([batch_dim, state_dim, state_dim], dtype=dtype)

    controls_state_jacobian_array = tf.TensorArray(dtype, size=horizon)

    value_hessian_array = tf.TensorArray(dtype, size=horizon)

    time = horizon - 1

    # run the planner forward through time

    lqr_results = tf.while_loop(
        lqr_condition,
        lqr_body, (
            initial_controls_state_jacobian,
            initial_value_hessian,
            dynamics_state_jacobian,
            dynamics_controls_jacobian,
            cost_state_hessian,
            cost_controls_hessian,
            controls_state_jacobian_array,
            value_hessian_array,
            time,
            horizon))

    # collect the outputs and make sure they are the correct shape

    controls_state_jacobian = tf.reshape(
        lqr_results[6].stack(),
        tf.concat([[horizon], batch_shape, [controls_dim, state_dim]], 0))

    value_hessian = tf.reshape(
        lqr_results[7].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, state_dim]], 0))

    return (
        controls_state_jacobian,
        value_hessian)
