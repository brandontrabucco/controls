"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.body import shooting_body
from controls.shooting.condition import shooting_condition
import tensorflow as tf


def shooting(
        initial_states,
        controls_state_jacobian,
        dynamics_model,
):
    """Solves for the value iteration solution to lqr.

    Args:
    - initial_states: the initial states from which to predict into the future
        with shape [batch_dim, ..., state_dim, 1].
    - controls_state_jacobian: the jacobian of the controls wrt. the states
        with shape [T, batch_dim, ..., controls_dim, state_dim].
    - dynamics_model: the dynamics as a tensorflow.keras.Model.

    Returns:
    - predicted_states: the next state with shape [T, batch_dim, ..., state_dim, 1].
    - controls: the controls with shape [batch_dim, controls_dim, 1].
    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the state
        with shape [T, batch_dim, ..., state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls
        with shape [T, batch_dim, ..., state_dim, controls_dim].
        """

    # get the batch shape and vector sizes

    horizon = tf.shape(controls_state_jacobian)[0]

    batch_shape = tf.shape(controls_state_jacobian)[1:-2]

    batch_dim = tf.reduce_prod(batch_shape)

    state_dim = tf.shape(controls_state_jacobian)[-1]

    controls_dim = tf.shape(controls_state_jacobian)[-1]

    dtype = controls_state_jacobian.dtype

    # check the horizon of every tensor

    tf.debugging.assert_equal(
        horizon,
        tf.shape(controls_state_jacobian)[0],
        message="controls_state_jacobian should have correct horizon")

    # make sure all inputs have the same batch shape

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(initial_states)[0:-2],
        message="initial_states should have correct batch shape")

    tf.debugging.assert_equal(
        batch_shape,
        tf.shape(controls_state_jacobian)[1:-2],
        message="controls_state_jacobian should have correct batch shape")

    # make sure all other dims are as expected

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(initial_states)[-2],
        message="initial_states should have shape [batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        1,
        tf.shape(initial_states)[-1],
        message="initial_states should have shape [batch_sim, ..., state_dim, state_dim]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(controls_state_jacobian)[-2],
        message="dynamics_controls_jacobian should have shape [T, batch_sim, ..., controls_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(controls_state_jacobian)[-1],
        message="dynamics_controls_jacobian should have shape [T, batch_sim, ..., controls_dim, state_dim]")

    # flatten the batch shape

    initial_states = tf.reshape(
        initial_states, [batch_dim, state_dim, 1])

    controls_state_jacobian = tf.reshape(
        controls_state_jacobian, [horizon, batch_dim, controls_dim, state_dim])

    # create the loop variables

    states_array = tf.TensorArray(dtype, size=horizon)

    controls_array = tf.TensorArray(dtype, size=horizon)

    dynamics_state_jacobian_array = tf.TensorArray(dtype, size=horizon)

    dynamics_controls_jacobian_array = tf.TensorArray(dtype, size=horizon)

    time = 0

    # run the planner forward through time

    shooting_results = tf.while_loop(
        shooting_condition,
        shooting_body, (
            initial_states,
            controls_state_jacobian,
            dynamics_model,
            states_array,
            controls_array,
            dynamics_state_jacobian_array,
            dynamics_controls_jacobian_array,
            time,
            horizon))

    # collect the outputs and make sure they are the correct shape

    states = tf.reshape(
        shooting_results[3].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

    controls = tf.reshape(
        shooting_results[4].stack(),
        tf.concat([[horizon], batch_shape, [controls_dim, 1]], 0))

    dynamics_state_jacobian = tf.reshape(
        shooting_results[5].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, state_dim]], 0))

    dynamics_controls_jacobian = tf.reshape(
        shooting_results[6].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, controls_dim]], 0))

    return (
        states,
        controls,
        dynamics_state_jacobian,
        dynamics_controls_jacobian)
