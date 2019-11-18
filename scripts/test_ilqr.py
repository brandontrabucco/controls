"""Author: Brandon Trabucco, Copyright 2019, MIT License"""

from controls.iterative_lqr import iterative_lqr
import tensorflow as tf

if __name__ == "__main__":

    A = tf.constant([[
        [-0.313, 56.7, 0.0],
        [-0.0139, -0.426, 0.0],
        [0.0, 56.7, 0.0]]])

    B = tf.constant([[
        [0.232],
        [0.0203],
        [0.0]]])

    states_inputs = tf.keras.layers.Input(shape=(3, 1))

    controls_inputs = tf.keras.layers.Input(shape=(1, 1))

    outputs = tf.keras.layers.Lambda(lambda x: tf.matmul(
        A, x[0]) + tf.matmul(B, x[1]))([states_inputs, controls_inputs])

    dynamics_model = tf.keras.Model(inputs=(states_inputs, controls_inputs), outputs=outputs)

    Q = tf.constant([[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]]])

    R = tf.constant([[
        [1.0]]])

    states_inputs = tf.keras.layers.Input(shape=(3, 1))

    controls_inputs = tf.keras.layers.Input(shape=(1, 1))

    outputs = tf.keras.layers.Lambda(lambda x: tf.matmul(
        tf.matmul(x[0], Q, transpose_a=True), x[0]) + tf.matmul(
            tf.matmul(x[1], R, transpose_a=True), x[1]))([states_inputs, controls_inputs])

    cost_model = tf.keras.Model(inputs=(states_inputs, controls_inputs), outputs=outputs)

    initial_states = tf.random.normal([1, 3, 1])

    results = iterative_lqr(
        initial_states,
        1,
        dynamics_model,
        cost_model,
        10,
        10)

    for i in range(10):

        controls = results[1][i, ...]

        costs = results[2][i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
