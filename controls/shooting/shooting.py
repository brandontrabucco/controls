"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.body import create_shooting_body
from controls.shooting.condition import shooting_condition
import tensorflow as tf


def shooting(
        x0,
        controls_model,
        dynamics_model,
        cost_model,
        h=5
):
    """Predicts into the future using shooting.

    Args:
    - x0: the initial states from which to predict into the future
        with shape [batch_dim, ..., state_dim, 1].

    - controls_model: the controls as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    - dynamics_model: the dynamics as a function.
        the function returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a function.
        the function returns tensors with shape [batch_dim, 1, 1].

    - h: the number of steps into the future for the planner.

    Returns:
    - xi: the states with shape [T, batch_dim, ..., state_dim, 1].
    - ui: the controls with shape [T, batch_dim, ..., controls_dim, 1].
    - ci: the costs with shape [T, batch_dim, ..., 1, 1].
    """
    shooting_results = tf.while_loop(
        shooting_condition,
        create_shooting_body(controls_model, dynamics_model, cost_model),
        (x0,
            tf.TensorArray(x0.dtype, size=h),
            tf.TensorArray(x0.dtype, size=h),
            tf.TensorArray(x0.dtype, size=h),
            0,
            h))

    return (
        shooting_results[1].stack(),
        shooting_results[2].stack(),
        shooting_results[3].stack())
