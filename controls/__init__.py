"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.cem import cem
from controls.shooting import shooting
from controls.iterative_lqr import iterative_lqr
from controls.lqr.lqr import lqr
from controls.distributions.gaussian import Gaussian
from controls.distributions.tanh_gaussian import TanhGaussian
from controls.distributions.deterministic import Deterministic
from controls.distributions.categorical import Categorical
from controls.distributions import UnitGaussian
from controls.distributions import UniformCategorical
