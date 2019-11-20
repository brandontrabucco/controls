"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.body import create_shooting_body
from controls.shooting.condition import shooting_condition
import tensorflow as tf


def shooting(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        horizon
):
    """Predicts into the future using shooting.

    Args:
    - initial_states: the initial states from which to predict into the future
        with shape [batch_dim, ..., state_dim, 1].
    - controls_model: the controls as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].

    - dynamics_model: the dynamics as a function.
        the function returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a function.
        the function returns tensors with shape [batch_dim, 1, 1].

    - horizon: the number of steps into the future for the planner.

    Returns:
    - states: the states with shape [T, batch_dim, ..., state_dim, 1].
    - controls: the controls with shape [T, batch_dim, ..., controls_dim, 1].
    - costs: the costs with shape [T, batch_dim, ..., 1, 1].
        """

    # get the batch shape and vector sizes

    batch_shape = tf.shape(initial_states)[:-2]

    batch_dim = tf.reduce_prod(batch_shape)

    state_dim = tf.shape(initial_states)[-2]

    dtype = initial_states.dtype

    # flatten the batch shape

    initial_states = tf.reshape(
        initial_states, [batch_dim, state_dim, 1])

    # create the loop variables

    states_array = tf.TensorArray(dtype, size=horizon)

    controls_array = tf.TensorArray(dtype, size=horizon)

    costs_array = tf.TensorArray(dtype, size=horizon)

    time = 0

    # run the planner forward through time

    shooting_results = tf.while_loop(
        shooting_condition,
        create_shooting_body(controls_model, dynamics_model, cost_model),
        (initial_states,
            states_array,
            controls_array,
            costs_array,
            time,
            horizon))

    # collect the outputs and make sure they are the correct shape

    states = tf.reshape(
        shooting_results[1].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

    controls = shooting_results[2].stack()

    controls_dim = tf.shape(controls)[-2]

    controls = tf.reshape(
        controls,
        tf.concat([[horizon], batch_shape, [controls_dim, 1]], 0))

    costs = tf.reshape(
        shooting_results[3].stack(),
        tf.concat([[horizon], batch_shape, [1, 1]], 0))

    return (
        states,
        controls,
        costs)
