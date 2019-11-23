"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def shooting_condition(
        x,
        x_array,
        u_array,
        c_array,
        time,
        horizon
):
    return tf.less(time, horizon)
