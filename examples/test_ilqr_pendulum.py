"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import iterative_lqr
from controls import shooting
import tensorflow as tf

from controls.distributions.gaussian import Gaussian

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

    controls_model = Gaussian(lambda time, inputs: (
        tf.zeros([1, 1]), tf.ones([1, 1, 1]), tf.ones([1, 1, 1]), tf.zeros([1])))

    def dynamics_model(time, inputs):
        return (A @ inputs[0][:, :, tf.newaxis] + B @ inputs[1][:, :, tf.newaxis])[:, :, 0]

    def cost_model(time, inputs):
        return 0.5 * (
            tf.matmul(tf.matmul(inputs[0][:, :, tf.newaxis], Q, transpose_a=True),
                      inputs[0][:, :, tf.newaxis]) +
            tf.matmul(tf.matmul(inputs[1][:, :, tf.newaxis], R, transpose_a=True),
                      inputs[1][:, :, tf.newaxis]))[:, 0, 0]

    initial_states = tf.random.normal([1, 4])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        n=10,
        a=0.0,
        random=False)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=20, random=False)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
