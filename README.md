# DiffOpt

DiffOpt implements differentiable optimal controls in TensorFlow 2.0. Have Fun! -Brandon

## Setup

Install this package using pip.

```
pip install git+git://github.com/brandontrabucco/controls.git
```

## Usage

Collect a batch of initial states to use for planning.

```python
# a tf.EagerTensor with shape [batch_dim, state_dim]
initial_states = tf.random.normal([1, 3])
```

Define an initial policy where the optimization process starts.

```python
# a diffopt.Distribution that returns [batch_dim, controls_dim]
controls_model = diffopt.Zeros(1)
```

Create a dynamics model that predicts future states given current states and controls.

```python
# a diffopt.Distribution that returns [batch_dim, state_dim]
A = tf.constant([[[-0.313, 56.7, 0.0], [-0.0139, -0.426, 0.0], [0.0, 56.7, 0.0]]])
B = tf.constant([[[0.232], [0.0203], [0.0]]])
dynamics_model = diffopt.Linear(0, [0, 0], [A, B])
```

Create a cost model that evaluates the cost of states and controls.

```python
# a diffopt.Distribution that returns [batch_dim, 1]
Q = tf.constant([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
R = tf.constant([[[1.0]]])
cost_model = diffopt.Quadratic(0, [0, 0], [0, 0], [[Q, 0], [0, R]])
```

Launch the optimizer to get a new policy.

```python
# a diffopt.TimeVaryingLinearGaussian that returns [batch_dim, controls_dim]
controls_model = diffopt.iterative_lqr(
    initial_states,
    controls_model,
    dynamics_model,
    cost_model,
    h=20,
    n=10,
    a=0.1,
    random=False)
```
