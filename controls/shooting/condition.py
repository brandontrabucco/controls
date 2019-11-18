"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def shooting_condition(
        states,
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
        horizon
):
    """Inner condition of shooting update loop."""

    # exit the loop if time is greater then or equal to horizon

    return tf.less(time, horizon)
