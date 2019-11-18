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
    - controls_model: the controls as a tensorflow.keras.Model.
        the model returns tensors with shape [batch_dim, controls_dim, 1].

    - dynamics_model: the dynamics as a tensorflow.keras.Model.
        the model returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a tensorflow.keras.Model.
        the model returns tensors with shape [batch_dim, 1, 1].

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

        - cost_shift: the shift term of the cost
            with shape [batch_dim, 1, 1].
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

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(states)

            controls = controls_model([states])

            tf.debugging.assert_equal(
                3,
                tf.size(tf.shape(controls)),
                message="controls should be a 3 tensor")

            tf.debugging.assert_equal(
                1,
                tf.shape(controls)[-1],
                message="controls should have shape [batch_dim, controls_dim, 1]")

            tape.watch(controls)

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

            cost_state_jacobian = tape.gradient(costs, states)

            cost_controls_jacobian = tape.gradient(costs, controls)

        # calculate the linearized dynamics

        dynamics_state_jacobian = tape.batch_jacobian(predicted_states, states)[:, :, 0, :, 0]

        dynamics_controls_jacobian = tape.batch_jacobian(predicted_states, controls)[:, :, 0, :, 0]

        dynamics_shift = predicted_states

        cost_state_state_hessian = tape.batch_jacobian(
            cost_state_jacobian, states)[:, :, 0, :, 0]

        cost_state_controls_hessian = tape.batch_jacobian(
            cost_state_jacobian, controls)[:, :, 0, :, 0]

        cost_controls_state_hessian = tape.batch_jacobian(
            cost_controls_jacobian, states)[:, :, 0, :, 0]

        cost_controls_controls_hessian = tape.batch_jacobian(
            cost_controls_jacobian, controls)[:, :, 0, :, 0]

        cost_shift = costs

        return (
            predicted_states,
            controls,
            costs,
            dynamics_state_jacobian,
            dynamics_controls_jacobian,
            dynamics_shift,
            cost_state_state_hessian,
            cost_state_controls_hessian,
            cost_controls_state_hessian,
            cost_controls_controls_hessian,
            cost_state_jacobian,
            cost_controls_jacobian,
            cost_shift)

    return shooting_update
