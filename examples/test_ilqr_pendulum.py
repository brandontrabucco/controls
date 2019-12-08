"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from diffopt import UnitGaussian
from diffopt import Linear
from diffopt import Quadratic
from diffopt import iterative_lqr
from diffopt import shooting
import tensorflow as tf


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

    controls_model = UnitGaussian(1)
    dynamics_model = Linear(0, [0, 0], [A, B])
    cost_model = Quadratic(0, [0, 0], [0, 0], [[Q, 0], [0, R]])

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
