"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from diffopt import UnitGaussian
from diffopt import Linear
from diffopt import Quadratic
from diffopt import cem
from diffopt import shooting
import tensorflow as tf


if __name__ == "__main__":

    A = tf.constant([[[-0.313, 56.7, 0.0],
                      [-0.0139, -0.426, 0.0],
                      [0.0, 56.7, 0.0]]])

    B = tf.constant([[[0.232],
                      [0.0203],
                      [0.0]]])

    Q = tf.constant([[[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]]])

    R = tf.constant([[[1.0]]])

    controls_model = UnitGaussian(1)
    dynamics_model = Linear(0, [0, 0], [A, B])
    cost_model = Quadratic(0, [0, 0], [0, 0], [[Q, 0], [0, R]])

    initial_states = tf.random.normal([1, 3])

    controls_model = cem(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        c=1000,
        n=100,
        k=100)

    xi, ui, ci = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=20, deterministic=False)

    for i in range(20):
        costs = ci[i, ...]
        print("Cost: {}".format(costs.numpy().sum()))
