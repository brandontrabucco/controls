"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.body import lqr_body
from controls.lqr.condition import lqr_condition
import tensorflow as tf


def lqr(
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
        horizon=10
):
    """Solves for the value iteration solution to lqr.

    Args:
    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the state
        with shape [batch_dim, ..., state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls
        with shape [batch_dim, ..., state_dim, controls_dim].

    - cost_state_hessian: the hessian of the cost wrt. the state
        with shape [batch_dim, ..., state_dim, state_dim].
    - cost_controls_hessian: the hessian of the cost wrt. the controls
        with shape [batch_dim, ..., controls_dim, controls_dim].

    - horizon: the number of bellman backups to perform.

    Returns:
    - controls_state_jacobians: the jacobians of the controls with respect to the states
        with shape [batch_dim, ..., controls_dim, state_dim].
    - value_hessians: the hessians of the cost to go wrt. the states
        with shape [batch_dim, ..., state_dim, state_dim].
        """

    # get the batch shape and vector sizes

    dtype = dynamics_controls_jacobian.dtype
    controls_dim = tf.shape(dynamics_controls_jacobian)[2]
    state_dim = tf.shape(dynamics_state_jacobian)[2]
    batch_shape = tf.shape(dynamics_state_jacobian)[:-2]
    batch_dim = tf.reduce_prod(batch_shape)

    # make sure all inputs have the same batch shape

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(dynamics_state_jacobian)[:-2],
        message="dynamics_state_jacobian should have correct batch shape")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(dynamics_controls_jacobian)[:-2],
        message="dynamics_controls_jacobian should have correct batch shape")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(cost_state_hessian)[:-2],
        message="cost_state_hessian should have correct batch shape")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(cost_controls_hessian)[:-2],
        message="cost_controls_hessian should have correct batch shape")

    # make sure all other dims are as expected

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-2],
        message="dynamics_state_jacobian should have shape [batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_state_jacobian)[-1],
        message="dynamics_state_jacobian should have shape [batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(dynamics_controls_jacobian)[-2],
        message="dynamics_controls_jacobian should have shape [batch_sim, ..., state_dim, controls_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(dynamics_controls_jacobian)[-1],
        message="dynamics_controls_jacobian should have shape [batch_sim, ..., state_dim, controls_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_hessian)[-2],
        message="cost_state_hessian should have shape [batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(cost_state_hessian)[-1],
        message="cost_state_hessian should have shape [batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_hessian)[-2],
        message="cost_controls_hessian should have shape [batch_sim, ..., controls_dim, controls_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(cost_controls_hessian)[-1],
        message="cost_controls_hessian should have shape [batch_sim, ..., controls_dim, controls_dim]")

    # flatten the batch shape

    dynamics_state_jacobian = tf.reshape(
        dynamics_state_jacobian, [batch_dim, state_dim, state_dim])

    dynamics_controls_jacobian = tf.reshape(
        dynamics_controls_jacobian, [batch_dim, state_dim, controls_dim])

    cost_state_hessian = tf.reshape(
        cost_state_hessian, [batch_dim, state_dim, state_dim])

    cost_controls_hessian = tf.reshape(
        cost_controls_hessian, [batch_dim, controls_dim, controls_dim])

    # create the loop variables

    initial_controls_state_jacobian = tf.zeros([batch_dim, controls_dim, state_dim], dtype=dtype)

    initial_value_hessian = tf.zeros([batch_dim, state_dim, state_dim], dtype=dtype)

    time = tf.constant(0)

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
            time,
            horizon))

    # collect the outputs and make sure they are the correct shape

    controls_state_jacobian = tf.reshape(
        lqr_results[0], tf.concat([batch_shape, [controls_dim, state_dim]], 0))

    value_hessian = tf.reshape(
        lqr_results[1], tf.concat([batch_shape, [state_dim, state_dim]], 0))

    return (
        controls_state_jacobian,
        value_hessian)
