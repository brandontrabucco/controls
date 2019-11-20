"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.lqr import lqr
from controls.shooting.shooting import shooting
from controls.time_varying_linear import time_varying_linear
from controls.taylor_series import first_order
from controls.taylor_series import second_order
import tensorflow as tf


def iterative_lqr(
        initial_states,
        controls_dim,
        dynamics_model,
        cost_model,
        horizon,
        num_iterations
):
    """Solves for the value iteration solution to lqr iteratively.

    Args:
    - initial_states: the initial states from which to predict into the future
        with shape [batch_dim, ..., state_dim, 1].
    - controls_dim: the cardinality of the controls variable.

    - dynamics_model: the dynamics as a function.
        the function returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a function.
        the function returns tensors with shape [batch_dim, 1, 1].

    - horizon: the number of steps into the future for the planner.
    - num_iterations: the number of iterations to run.

    Returns:
    - states: the states with shape [T, batch_dim, ..., state_dim, 1].
    - controls: the controls with shape [T, batch_dim, ..., controls_dim, 1].
    - costs: the costs with shape [T, batch_dim, ..., 1, 1].

    - dynamics_state_jacobian: the jacobian of the dynamics wrt. state i
        with shape [T, batch_dim, ..., state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. controls i
        with shape [T, batch_dim, ..., state_dim, controls_dim].

    - dynamics_shift: the shift term of the dynamics
        with shape [T, batch_dim, ..., state_dim, 1].

    - cost_state_state_hessian: the hessian of the cost wrt. state i state j
        with shape [T, batch_dim, ..., state_dim, state_dim].
    - cost_state_controls_hessian: the hessian of the cost wrt. state i controls j
        with shape [T, batch_dim, ..., state_dim, controls_dim].
    - cost_controls_state_hessian: the hessian of the cost wrt. controls i state j
        with shape [T, batch_dim, ..., controls_dim, state_dim].
    - cost_controls_controls_hessian: the hessian of the cost wrt. controls i controls j
        with shape [T, batch_dim, ..., controls_dim, controls_dim].

    - cost_state_jacobian: the jacobian of the cost wrt. state i
        with shape [T, batch_dim, ..., state_dim, 1].
    - cost_controls_jacobian: the jacobian of the cost wrt. controls i
        with shape [T, batch_dim, ..., controls_dim, 1].

    - cost_shift: the shift term of the cost
        with shape [batch_dim, 1, 1].
        """

    # get the batch shape and vector sizes

    batch_shape = tf.shape(initial_states)[:-2]

    batch_dim = tf.reduce_prod(batch_shape)

    state_dim = tf.shape(initial_states)[-2]

    # create the initial loop variables

    states = tf.zeros(
        tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

    controls = tf.zeros(
        tf.concat([[horizon], batch_shape, [controls_dim, 1]], 0))

    controls_state_jacobian = tf.zeros(
        tf.concat([[horizon], batch_shape, [controls_dim, state_dim]], 0))

    controls_shift = tf.zeros(
        tf.concat([[horizon], batch_shape, [controls_dim, 1]], 0))

    # iteratively run forward shooting and backward controls optimization with lqr

    for i in range(num_iterations):

        # update the controls model

        controls_model = time_varying_linear(
            states,
            controls + controls_shift,
            controls_state_jacobian)

        # run a forward pass using the shooting algorithm

        states, controls, costs = shooting(
            initial_states,
            controls_model,
            dynamics_model,
            cost_model,
            horizon)

        states = tf.reshape(
            states,
            [horizon * batch_dim, state_dim, 1])

        controls = tf.reshape(
            controls,
            [horizon * batch_dim, controls_dim, 1])

        # linear approximate the dynamics

        shifts, jacobians = first_order(
            dynamics_model,
            states,
            controls)

        dynamics_shift = tf.reshape(
            shifts,
            tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

        dynamics_state_jacobian = tf.reshape(
            jacobians[0],
            tf.concat([[horizon], batch_shape, [state_dim, state_dim]], 0))

        dynamics_controls_jacobian = tf.reshape(
            jacobians[1],
            tf.concat([[horizon], batch_shape, [state_dim, controls_dim]], 0))

        # quadratic approximate the dynamics

        shifts, jacobians, hessians = second_order(
            cost_model,
            states,
            controls)

        cost_state_jacobian = tf.reshape(
            jacobians[0],
            tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

        cost_controls_jacobian = tf.reshape(
            jacobians[1],
            tf.concat([[horizon], batch_shape, [controls_dim, 1]], 0))

        cost_state_state_hessian = tf.reshape(
            hessians[0][0],
            tf.concat([[horizon], batch_shape, [state_dim, state_dim]], 0))

        cost_state_controls_hessian = tf.reshape(
            hessians[1][0],
            tf.concat([[horizon], batch_shape, [state_dim, controls_dim]], 0))

        cost_controls_state_hessian = tf.reshape(
            hessians[0][1],
            tf.concat([[horizon], batch_shape, [controls_dim, state_dim]], 0))

        cost_controls_controls_hessian = tf.reshape(
            hessians[1][1],
            tf.concat([[horizon], batch_shape, [controls_dim, controls_dim]], 0))

        # run a backward pass using the linear quadratic regulator

        (controls_state_jacobian,
            controls_shift,
            value_state_state_hessian,
            value_state_jacobian) = lqr(
                dynamics_state_jacobian,
                dynamics_controls_jacobian,
                dynamics_shift,
                cost_state_state_hessian,
                cost_state_controls_hessian,
                cost_controls_state_hessian,
                cost_controls_controls_hessian,
                cost_state_jacobian,
                cost_controls_jacobian)

        states = tf.reshape(
            states,
            tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

        controls = tf.reshape(
            controls,
            tf.concat([[horizon], batch_shape, [controls_dim, 1]], 0))

    # update the controls model

    controls_model = time_varying_linear(
        states,
        controls + controls_shift,
        controls_state_jacobian)

    # run a forward pass using the shooting algorithm

    return shooting(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        horizon)
