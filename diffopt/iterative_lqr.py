"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from diffopt.lqr.lqr import lqr
from diffopt.shooting import shooting
from diffopt.taylor_series import first_order
from diffopt.taylor_series import second_order
from diffopt.distributions.linear import TimeVaryingLinearGaussian
import tensorflow as tf


def iterative_lqr(
        x0,
        controls_model,
        dynamics_model,
        cost_model,
        h=10,
        n=5,
        a=0.01,
        deterministic=True
):
    """Solves for the value iteration solution to lqr iteratively.

    Args:
    - x0: the initial states from which to predict into the future
        with shape [batch_dim, state_dim].

    - controls_model: the initial policy as a distribution.
        the function returns tensors with shape [batch_dim, controls_dim].
    - dynamics_model: the dynamics as a distribution.
        the function returns tensors with shape [batch_dim, state_dim].
    - cost_model: the cost as a distribution.
        the function returns tensors with shape [batch_dim, 1].

    - h: the number of steps into the future for the planner.
    - n: the number of iterations to run.
    - a: the weight of the cost function trust region.
    - deterministic: samples from the policy randomly if false.

    Returns:
    - controls_model: the policy as a function.
        the function returns tensors with shape [batch_dim, controls_dim].
    """
    xi, ui, ci = shooting(
        x0, controls_model, dynamics_model, cost_model, h=h, deterministic=deterministic)

    # collect the tensor shapes of the states and controls
    batch_dim = tf.shape(x0)[0]
    state_dim = tf.shape(x0)[1]
    controls_dim = tf.shape(ui)[2]

    # flatten and keep the last states visited
    xim1 = tf.reshape(xi, [h * batch_dim, state_dim])
    uim1 = tf.reshape(ui, [h * batch_dim, controls_dim])

    # run lqr iteratively for n steps
    for iteration in range(n):

        # run the forward dynamics probabilistically
        xi, ui, ci = shooting(
            x0, controls_model, dynamics_model, cost_model, h=h, deterministic=deterministic)

        # flatten the states and controls
        xi = tf.reshape(xi, [h * batch_dim, state_dim])
        ui = tf.reshape(ui, [h * batch_dim, controls_dim])

        # compute the first order taylor series
        Fx, Fu = first_order(dynamics_model, [xi, ui])[1:]

        # unflatten the taylor series
        Fx = tf.reshape(Fx, [h, batch_dim, state_dim, state_dim])
        Fu = tf.reshape(Fu, [h, batch_dim, state_dim, controls_dim])

        # wrap the cost model to make the hessian positive definite
        def wrapped_cost(time, inputs):
            x_error = (inputs[0] - xim1)[:, :, tf.newaxis]
            u_error = (inputs[1] - uim1)[:, :, tf.newaxis]
            return (1.0 - a) * cost_model(time, inputs) + a * (
                tf.matmul(x_error, x_error, transpose_a=True) +
                tf.matmul(u_error, u_error, transpose_a=True))[:, :, 0]

        # compute the second order taylor series
        Cx, Cu, Cxx, Cxu, Cux, Cuu = second_order(wrapped_cost, [xi, ui])[1:]

        # unflatten the taylor series
        Cx = tf.reshape(Cx, [h, batch_dim, state_dim])
        Cu = tf.reshape(Cu, [h, batch_dim, controls_dim])

        # unflatten the taylor series
        Cxx = tf.reshape(Cxx, [h, batch_dim, state_dim, state_dim])
        Cxu = tf.reshape(Cxu, [h, batch_dim, state_dim, controls_dim])
        Cux = tf.reshape(Cux, [h, batch_dim, controls_dim, state_dim])
        Cuu = tf.reshape(Cuu, [h, batch_dim, controls_dim, controls_dim])

        # perform lqr with the nonlinear models
        Qxx, Qxu, Qux, Quu, Qx, Qu, Kx, k, S, Vxx, Vx = lqr(
            Fx, Fu, Cxx, Cxu, Cux, Cuu, Cx, Cu)

        # save the last visited states
        xim1 = xi
        uim1 = ui

        # unflatten the states and controls
        inner_xi = tf.reshape(xi, [h, batch_dim, state_dim])
        inner_ui = tf.reshape(ui, [h, batch_dim, controls_dim])

        # create a new linear gaussian policy
        controls_model = TimeVaryingLinearGaussian(
            inner_ui + k, [inner_xi], S, [Kx])

    return controls_model
