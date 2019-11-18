"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def lqr_condition(
        controls_state_jacobian,
        controls_shift,
        value_state_state_hessian,
        value_state_jacobian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_shift,
        cost_state_state_hessian,
        cost_state_controls_hessian,
        cost_controls_state_hessian,
        cost_controls_controls_hessian,
        cost_state_jacobian,
        cost_controls_jacobian,
        controls_state_jacobian_array,
        controls_shift_array,
        value_state_state_hessian_array,
        value_state_jacobian_array,
        time,
        horizon
):
    """Inner condition of lqr update loop."""

    # exit the loop if time is less than zero

    return tf.greater_equal(time, 0)
