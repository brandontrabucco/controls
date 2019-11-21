"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def create_shooting_update(
        controls_model,
        dynamics_model,
        cost_model
):
    """Predicts into the future using a policy and dynamics model.

    Args:
    - states: the current state with shape [batch_dim, state_dim, 1].
    - controls_model: the controls as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].

    - dynamics_model: the dynamics as function.
        the model returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a function.
        the function returns tensors with shape [batch_dim, 1, 1].

    Returns:
    - a function shooting_update
    """

    def shooting_update(
            states,
    ):
        """Predicts into the future using a policy and dynamics model.

        Args:
        - states: the current state with shape [batch_dim, state_dim, 1].

        Returns:
        - predicted_states: the next state with shape [batch_dim, state_dim, 1].
        - controls: the controls with shape [batch_dim, controls_dim, 1].
        - costs: the costs with shape [batch_dim, 1, 1].
        """

        tf.debugging.assert_equal(
            3,
            tf.size(tf.shape(states)),
            message="states should be a 3 tensor")

        tf.debugging.assert_equal(
            1,
            tf.shape(states)[-1],
            message="states should have shape [batch_dim, state_dim, 1]")

        # calculate the controls and the next state using the dynamics

        controls = controls_model([states])

        tf.debugging.assert_equal(
            3,
            tf.size(tf.shape(controls)),
            message="controls should be a 3 tensor")

        tf.debugging.assert_equal(
            1,
            tf.shape(controls)[-1],
            message="controls should have shape [batch_dim, controls_dim, 1]")

        predicted_states = dynamics_model([states, controls])

        tf.debugging.assert_equal(
            3,
            tf.size(tf.shape(predicted_states)),
            message="predicted_states should be a 3 tensor")

        tf.debugging.assert_equal(
            1,
            tf.shape(predicted_states)[-1],
            message="predicted_states should have shape [batch_dim, state_dim, 1]")

        costs = cost_model([states, controls])

        tf.debugging.assert_equal(
            3,
            tf.size(tf.shape(costs)),
            message="costs should be a 3 tensor")

        tf.debugging.assert_equal(
            [1, 1],
            tf.shape(costs)[1:],
            message="costs should have shape [batch_dim, 1, 1]")

        return (
            predicted_states,
            controls,
            costs)

    return shooting_update
