"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def lqr_condition(
        controls_state_jacobian,
        value_hessian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
        controls_state_jacobian_array,
        value_hessian_array,
        time,
        horizon
):
    """Inner condition of lqr update loop."""

    # exit the loop if time is less than zero

    return tf.greater_equal(time, 0)
