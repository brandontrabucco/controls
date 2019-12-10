"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def shooting(
        x0,
        controls_model,
        dynamics_model,
        cost_model,
        h=5,
        deterministic=True
):
    """Predicts into the future using random shooting.

    Args:
    - x0: the initial states from which to predict into the future
        with shape [batch_dim, state_dim].

    - controls_model: the controls as a random function.
        the function returns tensors with shape [batch_dim, controls_dim].
    - dynamics_model: the dynamics as a random function.
        the function returns tensors with shape [batch_dim, state_dim].
    - cost_model: the cost as a random function.
        the function returns tensors with shape [batch_dim].

    - h: the number of steps into the future for the planner.
    - deterministic: samples from the policy randomly if false.

    Returns:
    - xi: the states with shape [T, batch_dim, state_dim].
    - ui: the controls with shape [T, batch_dim, controls_dim].
    - ci: the costs with shape [T, batch_dim].
    """
    def shooting_body(
        x,
        x_array,
        u_array,
        c_array,
        time,
        horizon
    ):
        u = (controls_model.expected_value(
            time, [x]) if deterministic else controls_model.sample(time, [x]))[0]

        return (
            dynamics_model(time, [x, u]),
            x_array.write(time, x),
            u_array.write(time, u),
            c_array.write(time, cost_model(time, [x, u])),
            time + 1,
            horizon)

    u0 = controls_model(0, [x0])
    shooting_results = tf.while_loop(
        lambda x, x_array, u_array, c_array, time, horizon: tf.less(time, horizon),
        shooting_body,
        (x0,
            tf.TensorArray(x0.dtype, size=h),
            tf.TensorArray(u0.dtype, size=h),
            tf.TensorArray(tf.float32, size=h),
            0,
            h))

    return (
        shooting_results[1].stack(),
        shooting_results[2].stack(),
        shooting_results[3].stack())
