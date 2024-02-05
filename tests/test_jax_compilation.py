"""
Tests for bayesian_ODE package.
"""
import pytest

# import jax
import pymc as pm
import jax
import jax.numpy as jnp
from pytensor.tensor.type import TensorType

from bayesian_ODE import create_and_register_jax


@pytest.fixture
def test_models():
    # 2 parameters input, tuple output
    with pm.Model() as model1:
        x, y = pm.Normal("input", size=2)

        def f(x, y):
            return jax.nn.sigmoid(x + y), y * 2

        f_op = create_and_register_jax(f)
        out, _ = f_op(x, y)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input, single output
    with pm.Model() as model2:
        x, y = pm.Normal("input", size=2)

        def f2(x, y):
            return jax.nn.sigmoid(x + y)

        f2_op = create_and_register_jax(f2, output_shape_func=lambda x, y: x)
        out = f2_op(x, y)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input, list output
    with pm.Model() as model3:
        x, y = pm.Normal("input", size=2)

        def f(x, y):
            return [jax.nn.sigmoid(x + y), y * 2]

        f_op = create_and_register_jax(
            f,
            output_shape_func=lambda x, y: [x, y],
        )
        out, _ = f_op(x, y)
        pm.Normal("obs", out, observed=3)

    # single 1d input, tuple output
    with pm.Model() as model4:
        x, y = pm.Normal("input", size=2)
        print(x.shape)

        def f4(x):
            return jax.nn.sigmoid(x), x * 2

        f4_op = create_and_register_jax(f4, output_shape_func=lambda x: (x, x))
        out, _ = f4_op(x)
        pm.Normal("obs", out, observed=3)

    # single 0d input, tuple output
    with pm.Model() as model5:
        x = pm.Normal("input", size=())

        def f5(x):
            return jax.nn.sigmoid(x), x

        f5_op = create_and_register_jax(f5, output_shape_func=lambda x: (x, x))
        out, _ = f5_op(x)
        pm.Normal("obs", out, observed=3)

    # single input, list output
    with pm.Model() as model6:
        x, y = pm.Normal("input", size=2)

        def f(x):
            return [jax.nn.sigmoid(x), 2 * x]

        f_op = create_and_register_jax(
            f,
            output_shape_func=lambda x: [x, x],
        )
        out, _ = f_op(x)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input with pytree, tuple output
    with pm.Model() as model7:
        x, y = pm.Normal("input", size=2)
        y_tmp = {"y": y, "y2": [y**2]}

        def f(x, y):
            return jax.nn.sigmoid(x), 2 * x + y["y"] + y["y2"][0]

        f_op = create_and_register_jax(f, output_shape_func=lambda x, y: (x, x))
        out, _ = f_op(x, y_tmp)
        pm.Normal("obs", out, observed=3)

    # 2 parameters input with pytree, pytree output
    with pm.Model() as model8:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=(1,))
        y_tmp = {"y3": y, "y4": [y**2]}

        def f(x, y):
            jax.debug.print("x: {}", x)
            jax.debug.print("y: {}", y)
            return x, jax.tree_map(
                lambda x: jnp.exp(x), y
            )  # {"a": y["y4"], "b": y["y3"]}  # , jax.tree_map(jnp.exp, y)

        f_op = create_and_register_jax(
            f,
            output_shape_func=lambda x, y: (x, y)  # (x, {"a": [()], "b": ()})  # , y),
            # args_for_graph=["x", "y"],
        )
        out_x, out_y = f_op(x, y_tmp)
        # for_model = out_y["y3"]
        pm.Normal("obs", out_x, observed=(3, 2, 3))
        # pm.Normal("obs2", out_y["y3"], observed=(3,))

    # 2 parameters input with pytree, pytree output
    with pm.Model() as model9:
        x = pm.Normal("input", size=3)
        y = pm.Normal("input2", size=1)
        y_tmp = {"y3": y, "y4": [y**2]}

        def f(x, y, non_model_arg):
            print(non_model_arg)
            return x, jax.tree_map(jnp.exp, y)

        f_op = create_and_register_jax(
            f,
            output_shape_func=lambda x, y: (x, y),
            args_for_graph=["x", "y"],
            static_argnums=(2,),
        )
        out_x, out_y = f_op(x, y_tmp, "Hello World!")
        # raise RuntimeError()
        # for_model = out_y["y3"]
        pm.Normal("obs", out_x, observed=(3, 2, 3))

    return model1, model2, model3, model4, model5, model6, model7, model8  # , model9


def test_jax_compilation(test_models):
    for i, model in enumerate(test_models):
        print(f"Test model {i + 1}")

        ip = model.initial_point()
        # Setting the mode to fast_compile shouldn't make a difference in the test coverage
        logp_fn = model.compile_fn(model.logp(sum=False), mode="FAST_COMPILE")
        logp_fn(ip)
        dlogp_fn = model.compile_fn(model.dlogp(), mode="FAST_COMPILE")
        dlogp_fn(ip)

        ip = model.initial_point()
        logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
        logp_fn(ip)
        dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
        dlogp_fn(ip)


def test_jax_without_jit(test_models):
    with jax.disable_jit():
        for i, model in enumerate(test_models):
            print(f"Test model {i + 1}")

            ip = model.initial_point()
            logp_fn = model.compile_fn(model.logp(sum=False), mode="JAX")
            logp_fn(ip)
            dlogp_fn = model.compile_fn(model.dlogp(), mode="JAX")
            dlogp_fn(ip)
