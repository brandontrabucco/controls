"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def shooting_update(
        states,
        controls_state_jacobian,
        dynamics_model,
):
    """Predicts into the future using a policy and dynamics model.

    Args:
    - states: the current state with shape [batch_dim, state_dim, 1].
    - controls_state_jacobian: the jacobian of the controls with respect to the state
        with shape [batch_dim, controls_dim, state_dim].
    - dynamics_model: the dynamics as a tensorflow.keras.Model.

    Returns:
    - predicted_states: the next state with shape [batch_dim, state_dim, 1].
    - controls: the controls with shape [batch_dim, controls_dim, 1].
    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the state
        with shape [batch_dim, state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls
        with shape [batch_dim, state_dim, controls_dim].
        """

    # get the batch shape and vector sizes

    controls_dim = tf.shape(controls_state_jacobian)[-2]

    state_dim = tf.shape(states)[-2]

    batch_dim = tf.shape(states)[0]

    # make sure all inputs are 3 tensors

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(states)),
        message="states should be a 3 tensor")

    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(controls_state_jacobian)),
        message="controls_state_jacobian should be a 3 tensor")

    # make sure all inputs have the same batch shape

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(states)[0],
        message="states should have correct batch size")

    tf.debugging.assert_equal(
        batch_dim,
        tf.shape(controls_state_jacobian)[0],
        message="controls_state_jacobian should have correct batch size")

    # make sure all other dims are as expected

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(states)[-2],
        message="states should have shape [batch_sim, state_dim, 1]")

    tf.debugging.assert_equal(
        1,
        tf.shape(states)[-1],
        message="states should have shape [batch_sim, state_dim, 1]")

    tf.debugging.assert_equal(
        controls_dim,
        tf.shape(controls_state_jacobian)[-2],
        message="controls_state_jacobian should have shape [batch_sim, controls_dim, state_dim]")

    tf.debugging.assert_equal(
        state_dim,
        tf.shape(controls_state_jacobian)[-1],
        message="controls_state_jacobian should have shape [batch_sim, controls_dim, state_dim]")

    # calculate the controls and the next state using the dynamics

    with tf.GradientTape(persistent=True) as tape:

        controls = tf.matmul(controls_state_jacobian, states)

        predicted_states = dynamics_model([states, controls])

    # calculate the linearized dynamics

    dynamics_state_jacobian = tape.batch_jabocian(predicted_states, states)[:, :, 0, :, 0]

    dynamics_controls_jacobian = tape.batch_jabocian(predicted_states, controls)[:, :, 0, :, 0]

    return (
        predicted_states,
        controls,
        dynamics_state_jacobian,
        dynamics_controls_jacobian)
