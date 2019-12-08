"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from diffopt.cem import cem
from diffopt.shooting import shooting
from diffopt.iterative_lqr import iterative_lqr
from diffopt.lqr.lqr import lqr

from diffopt.distributions.distribution import Distribution
from diffopt.distributions.continuous.gaussian import Gaussian
from diffopt.distributions.discrete.categorical import Categorical
from diffopt.distributions.continuous.deterministic import Deterministic

from diffopt.distributions import Zeros
from diffopt.distributions import UnitGaussian
from diffopt.distributions import UniformCategorical

from diffopt.distributions.linear import LinearGaussian
from diffopt.distributions.linear import TimeVaryingLinearGaussian
from diffopt.distributions.linear import Linear
from diffopt.distributions.linear import TimeVaryingLinear

from diffopt.distributions.quadratic import QuadraticGaussian
from diffopt.distributions.quadratic import TimeVaryingQuadraticGaussian
from diffopt.distributions.quadratic import Quadratic
from diffopt.distributions.quadratic import TimeVaryingQuadratic
