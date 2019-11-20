"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def shooting_condition(
        states,
        states_array,
        controls_array,
        costs_array,
        time,
        horizon
):
    """Inner condition of shooting update loop."""

    # exit the loop if time is greater then or equal to horizon

    return tf.less(time, horizon)
