"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import iterative_lqr
from controls import shooting
import tensorflow as tf

from controls.distributions.deterministic import Deterministic

if __name__ == "__main__":

    A = tf.constant([[[0., 1., 0., 0.],
                      [0., -.1818, 2.6727, 0.],
                      [0., 0., 0., 1.],
                      [0., -.4545, 31.1818, 0.]]])

    B = tf.constant([[[0.],
                      [1.8182],
                      [0.],
                      [4.5455]]])

    Q = tf.constant([[[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]]])

    R = tf.constant([[[1.0]]])

    controls_model = Deterministic(lambda time, inputs: tf.zeros([1, 1, 1]))
    dynamics_model = Deterministic(lambda time, inputs: A @ inputs[0] + B @ inputs[1])
    cost_model = Deterministic(lambda time, inputs: 0.5 * (
        tf.matmul(tf.matmul(inputs[0], Q, transpose_a=True), inputs[0]) +
        tf.matmul(tf.matmul(inputs[1], R, transpose_a=True), inputs[1])))

    initial_states = tf.random.normal([1, 4, 1])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        n=10,
        a=0.0)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=20)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
