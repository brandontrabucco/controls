"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.lqr import lqr
from controls.shooting import shooting
from controls.taylor_series import first_order
from controls.taylor_series import second_order
from controls.distributions.gaussian import Gaussian
import tensorflow as tf


def iterative_lqr(
        x0,
        controls_model,
        dynamics_model,
        cost_model,
        h=10,
        n=5,
        a=0.01,
        random=True
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
        the function returns tensors with shape [batch_dim].

    - h: the number of steps into the future for the planner.
    - n: the number of iterations to run.
    - a: the weight of the cost function trust region.
    - random: samples from the policy randomly if true.

    Returns:
    - controls_model: the policy as a function.
        the function returns tensors with shape [batch_dim, controls_dim].
    """
    xi, ui, ci = shooting(x0, controls_model, dynamics_model, cost_model, h=h, random=random)

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
        xi, ui, ci = shooting(x0, controls_model, dynamics_model, cost_model, h=h, random=random)

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
            trust_region = (
                tf.matmul(x_error, x_error, transpose_a=True) +
                tf.matmul(u_error, u_error, transpose_a=True))[:, 0, 0]
            return ((1.0 - a) * cost_model(
                time, inputs) + a * trust_region)[:, tf.newaxis]

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

        # build the parameters of a linear gaussian
        def get_parameters(time, inputs):
            x_delta = inputs[0] - inner_xi[time, :, :]
            u_delta = (Kx[time, :, :, :] @ x_delta[:, :, tf.newaxis])[:, :, 0] + k[time, :, :]
            mean = inner_ui[time, :, :] + u_delta
            covariance = S[time, :, :, :]
            return (mean,
                    tf.linalg.sqrtm(covariance),
                    tf.linalg.inv(covariance),
                    tf.linalg.logdet(covariance))

        # create a new linear gaussian policy
        controls_model = Gaussian(get_parameters)

    return controls_model
