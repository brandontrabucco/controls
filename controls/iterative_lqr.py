"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.lqr import lqr
from controls.shooting.shooting import shooting
from controls.time_varying_linear import time_varying_linear
from controls.taylor_series import first_order
from controls.taylor_series import second_order
import tensorflow as tf


def iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        horizon=10,
        num_iterations=5,
        trust_region_alpha=0.01,
):
    """Solves for the value iteration solution to lqr iteratively.

    Args:
    - initial_states: the initial states from which to predict into the future
        with shape [batch_dim, state_dim, 1].

    - controls_model: the initial policy as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    - dynamics_model: the dynamics as a function.
        the function returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a function.
        the function returns tensors with shape [batch_dim, 1, 1].

    - horizon: the number of steps into the future for the planner.
    - num_iterations: the number of iterations to run.
    - trust_region_alpha: the weight of the cost function trust region.

    Returns:
    - controls_model: the initial policy as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    """

    # check that all inputs are 3 tensors
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(initial_states)),
        message="initial_states should be a 3 tensor")

    # get the batch shape and vector sizes
    batch_dim = tf.shape(initial_states)[0]
    state_dim = tf.shape(initial_states)[1]
    dtype = initial_states.dtype

    # run an initial forward pass using the shooting algorithm
    states, controls, costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, horizon)

    # infer the cardinality of the controls from the shooting
    controls_dim = tf.shape(controls)[2]

    last_states = tf.reshape(states, [horizon * batch_dim, state_dim, 1])
    last_controls = tf.reshape(controls, [horizon * batch_dim, controls_dim, 1])

    # iteratively run forward shooting and backward controls optimization with lqr
    for i in range(num_iterations):

        # run a forward pass using the shooting algorithm
        states, controls, costs = shooting(
            initial_states, controls_model, dynamics_model, cost_model, horizon)

        states = tf.reshape(states, [horizon * batch_dim, state_dim, 1])
        controls = tf.reshape(controls, [horizon * batch_dim, controls_dim, 1])

        # linear approximate the dynamics
        shifts, jacobians = first_order(dynamics_model, [states, controls])

        dynamics_state_jacobian = tf.reshape(
            jacobians[0], [horizon, batch_dim, state_dim, state_dim])
        dynamics_controls_jacobian = tf.reshape(
            jacobians[1], [horizon, batch_dim, state_dim, controls_dim])

        # quadratic approximate the cost function with a trust region
        def trust_region_cost_model(x):
            return (1.0 - trust_region_alpha) * cost_model(x) + trust_region_alpha * (
                tf.matmul(x[0] - last_states, x[0] - last_states, transpose_a=True) +
                tf.matmul(x[1] - last_controls, x[1] - last_controls, transpose_a=True))

        shifts, jacobians, hessians = second_order(trust_region_cost_model, [states, controls])

        cost_state_jacobian = tf.reshape(jacobians[0], [horizon, batch_dim, state_dim, 1])
        cost_controls_jacobian = tf.reshape(jacobians[1], [horizon, batch_dim, controls_dim, 1])

        cost_state_state_hessian = tf.reshape(
            hessians[0][0], [horizon, batch_dim, state_dim, state_dim])
        cost_state_controls_hessian = tf.reshape(
            hessians[1][0], [horizon, batch_dim, state_dim, controls_dim])
        cost_controls_state_hessian = tf.reshape(
            hessians[0][1], [horizon, batch_dim, controls_dim, state_dim])
        cost_controls_controls_hessian = tf.reshape(
            hessians[1][1], [horizon, batch_dim, controls_dim, controls_dim])

        # run a backward pass using the linear quadratic regulator
        controls_state_jacobian, controls_shift, value_state_state_hessian, value_state_jacobian = lqr(
                dynamics_state_jacobian,
                dynamics_controls_jacobian,
                tf.zeros([horizon, batch_dim, state_dim, 1], dtype=dtype),
                cost_state_state_hessian,
                cost_state_controls_hessian,
                cost_controls_state_hessian,
                cost_controls_controls_hessian,
                cost_state_jacobian,
                cost_controls_jacobian)

        last_states = states
        last_controls = controls
        states = tf.reshape(states, [horizon, batch_dim, state_dim, 1])
        controls = tf.reshape(controls, [horizon, batch_dim, controls_dim, 1])

        # update the controls model
        controls_model = time_varying_linear(
            controls + controls_shift, [states], [controls_state_jacobian])

    # return the latest and greatest controls model

    return controls_model
