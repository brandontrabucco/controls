"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import Zeros
from controls import Linear
from controls import Quadratic
from controls import iterative_lqr
from controls import shooting
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

    controls_model = Zeros(1)
    dynamics_model = Linear(0, [0, 0], [A, B])
    cost_model = Quadratic(0, [0, 0], [0, 0], [[Q, 0], [0, R]])

    initial_states = tf.random.normal([1, 3])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        n=10,
        a=0.1,
        random=False)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=20, random=False)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
