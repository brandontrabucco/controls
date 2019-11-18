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
    - controls_model: the controls as a tensorflow.keras.Model.
        the model returns tensors with shape [batch_dim, controls_dim, 1].

    - dynamics_model: the dynamics as a tensorflow.keras.Model.
        the model returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a tensorflow.keras.Model.
        the model returns tensors with shape [batch_dim, 1, 1].

    - horizon: the number of steps into the future for the planner.

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

    batch_shape = tf.shape(initial_states)[1:-2]

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

    dynamics_state_jacobian_array = tf.TensorArray(dtype, size=horizon)

    dynamics_controls_jacobian_array = tf.TensorArray(dtype, size=horizon)

    dynamics_shift_array = tf.TensorArray(dtype, size=horizon)

    cost_state_state_hessian_array = tf.TensorArray(dtype, size=horizon)

    cost_state_controls_hessian_array = tf.TensorArray(dtype, size=horizon)

    cost_controls_state_hessian_array = tf.TensorArray(dtype, size=horizon)

    cost_controls_controls_hessian_array = tf.TensorArray(dtype, size=horizon)

    cost_state_jacobian_array = tf.TensorArray(dtype, size=horizon)

    cost_controls_jacobian_array = tf.TensorArray(dtype, size=horizon)

    cost_shift_array = tf.TensorArray(dtype, size=horizon)

    time = 0

    # run the planner forward through time

    shooting_results = tf.while_loop(
        shooting_condition,
        create_shooting_body(
            controls_model,
            dynamics_model,
            cost_model
        ), (
            initial_states,
            states_array,
            controls_array,
            costs_array,
            dynamics_state_jacobian_array,
            dynamics_controls_jacobian_array,
            dynamics_shift_array,
            cost_state_state_hessian_array,
            cost_state_controls_hessian_array,
            cost_controls_state_hessian_array,
            cost_controls_controls_hessian_array,
            cost_state_jacobian_array,
            cost_controls_jacobian_array,
            cost_shift_array,
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

    dynamics_state_jacobian = tf.reshape(
        shooting_results[4].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, state_dim]], 0))

    dynamics_controls_jacobian = tf.reshape(
        shooting_results[5].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, controls_dim]], 0))

    dynamics_shift = tf.reshape(
        shooting_results[6].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

    cost_state_state_hessian = tf.reshape(
        shooting_results[7].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, state_dim]], 0))

    cost_state_controls_hessian = tf.reshape(
        shooting_results[8].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, controls_dim]], 0))

    cost_controls_state_hessian = tf.reshape(
        shooting_results[9].stack(),
        tf.concat([[horizon], batch_shape, [controls_dim, state_dim]], 0))

    cost_controls_controls_hessian = tf.reshape(
        shooting_results[10].stack(),
        tf.concat([[horizon], batch_shape, [controls_dim, controls_dim]], 0))

    cost_state_jacobian = tf.reshape(
        shooting_results[11].stack(),
        tf.concat([[horizon], batch_shape, [state_dim, 1]], 0))

    cost_controls_jacobian = tf.reshape(
        shooting_results[12].stack(),
        tf.concat([[horizon], batch_shape, [controls_dim, 1]], 0))

    cost_shift = tf.reshape(
        shooting_results[13].stack(),
        tf.concat([[horizon], batch_shape, [1, 1]], 0))

    return (
        states,
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
