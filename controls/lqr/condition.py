"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def lqr_condition(
        Vxx,
        Vx,
        Fx,
        Fu,
        Cxx,
        Cxu,
        Cux,
        Cuu,
        Cx,
        Cu,
        Qxx_array,
        Qxu_array,
        Qux_array,
        Quu_array,
        Qx_array,
        Qu_array,
        Kx_array,
        k_array,
        Vxx_array,
        Vx_array,
        time,
        horizon
):
    return tf.greater_equal(time, 0)
