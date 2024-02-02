"""
Tests for bayesian_ODE package.
"""
import pytest
import pymc as pm
import jax
from pytensor.tensor.type import TensorType

from bayesian_ODE import create_and_register_jax


@pytest.fixture
def test_models():
    with pm.Model() as model1:
        x, y = pm.Normal("input", size=2)

        def f(x, y):
            return jax.nn.sigmoid(x + y), y * 2

        f_op = create_and_register_jax(
            f, output_types=[TensorType(dtype="float64", shape=()) for _ in range(2)]
        )
        out, _ = f_op(x, y)
        pm.Normal("obs", out, observed=3)

    with pm.Model() as model2:
        x, y = pm.Normal("input", size=2)

        def f2(x, y):
            return jax.nn.sigmoid(x + y)

        f2_op = create_and_register_jax(
            f2, output_types=[TensorType(dtype="float64", shape=()) for _ in range(1)]
        )
        out = f2_op(x, y)
        pm.Normal("obs", out, observed=3)

    with pm.Model() as model3:
        x, y = pm.Normal("input", size=2)

        def f(x, y):
            return [jax.nn.sigmoid(x + y), y * 2]

        f_op = create_and_register_jax(
            f, output_types=[TensorType(dtype="float64", shape=()) for _ in range(2)]
        )
        out, _ = f_op(x, y)
        pm.Normal("obs", out, observed=3)

    with pm.Model() as model4:
        x, y = pm.Normal("input", size=2)
        print(x.shape)

        def f4(x):
            return jax.nn.sigmoid(x), x * 2

        f4_op = create_and_register_jax(
            f4, output_types=[TensorType(dtype="float64", shape=()) for _ in range(2)]
        )
        out, _ = f4_op(x)
        pm.Normal("obs", out, observed=3)

    with pm.Model() as model5:
        x = pm.Normal("input", size=())

        def f5(x):
            return jax.nn.sigmoid(x), x

        f5_op = create_and_register_jax(
            f5, output_types=[TensorType(dtype="float64", shape=()) for _ in range(2)]
        )
        out, _ = f5_op(x)
        pm.Normal("obs", out, observed=3)

    with pm.Model() as model6:
        x, y = pm.Normal("input", size=2)

        def f(x):
            return [jax.nn.sigmoid(x), 2 * x]

        f_op = create_and_register_jax(
            f, output_types=[TensorType(dtype="float64", shape=()) for _ in range(2)]
        )
        out, _ = f_op(x)
        pm.Normal("obs", out, observed=3)

    with pm.Model() as model7:
        x, y = pm.Normal("input", size=2)
        y_tmp = {"y": y, "y2": [y**2]}

        def f(x, y):
            return jax.nn.sigmoid(x), 2 * x + y["y"] + y["y2"][0]

        f_op = create_and_register_jax(
            f, output_types=[TensorType(dtype="float64", shape=()) for _ in range(2)]
        )
        out, _ = f_op(x, y_tmp)
        pm.Normal("obs", out, observed=3)
    return model1, model2, model3, model4, model5, model6, model7


def test_jax_compilation(test_models):
    for i, model in enumerate(test_models):
        print(f"model {i + 1}")

        ip = model.initial_point()
        logp_fn = model.compile_fn(model.logp(sum=False), mode="FAST_COMPILE")
        logp_fn(ip)
        dlogp_fn = model.compile_fn(model.dlogp(), mode="FAST_COMPILE")
        dlogp_fn(ip)
        print("Timing C compiled:")
        logp_fn(ip)
        dlogp_fn(ip)

        ip = model.initial_point()
        logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
        logp_fn(ip)
        dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
        dlogp_fn(ip)
        print("Timing JAX compiled:")
        logp_fn(ip)
        dlogp_fn(ip)
