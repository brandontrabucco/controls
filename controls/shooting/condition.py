"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def shooting_condition(
        states,
        controls_state_jacobian,
        dynamics_model,
        states_array,
        controls_array,
        dynamics_state_jacobian_array,
        dynamics_controls_jacobian_array,
        time,
        horizon
):
    """Inner condition of shooting update loop."""

    # exit the loop if time is greater then or equal to horizon

    return tf.less(time, horizon)
