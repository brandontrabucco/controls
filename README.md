# Controls

This package implements differentiable optimal controls in TensorFlow 2.0. Have Fun! -Brandon

# Setup

Install this package directly from github using pip.

```
pip install git+git://github.com/brandontrabucco/controls.git
```

# Usage

Import the control algorithm you want to use.

```
from controls import UnitGaussian
from controls import Linear
from controls import Quadratic
from controls import cem
import tensorflow as tf
```

Collect a batch of initial states.

```
initial_states = tf.random.normal([1, 3])
```

Define your dynamics model and cost model.

```
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

dynamics_model = Linear(0, [0, 0], [A, B])
cost_model = Quadratic(0, [0, 0], [0, 0], [[Q, 0], [0, R]])
```

Define your initial policy where the optimizer starts.

```
controls_model = UnitGaussian(1)
```

Launch the optimizer to get a new policy.

```
controls_model = cem(
    initial_states,
    controls_model,
    dynamics_model,
    cost_model,
    h=20,
    c=1000,
    n=100,
    k=100)
```
