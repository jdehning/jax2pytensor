"""Convert a jax function to a pytensor compatible function."""

import inspect
from collections.abc import Sequence
import logging

import jax
import numpy as np
import pytensor.tensor as pt
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
import jax.experimental.export.shape_poly


log = logging.getLogger(__name__)


class UnknDim(int):
    def __new__(cls):
        return super(cls, cls).__new__(cls, 1)

    def __add__(self, other):
        return self.__class__()

    def __sub__(self, other):
        return self.__class__()

    def __mul__(self, other):
        return self.__class__()

    def __div__(self, other):
        return self.__class__()


def jax2pytensor(
    jaxfunc,
    args_for_graph="all",
    output_formatter=None,
    name=None,
    static_argnames=(),
):
    """Return a pytensor from a jax jittable function.

    It requires to define the output types of the returned values as pytensor types. A
    unique name should also be passed in case the name of the jaxfunc is identical to
    some other node. The design of this function is based on
    https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick-examples/

    Parameters
    ----------
    jaxfunc : jax jittable function
        function for which the node is created, can return multiple tensors as a tuple.
    args_for_graph : list of str or "all"
        If "all", all arguments except arguments passed via **kwargs are used for the graph.
        Otherwise specify a list of argument names to use for the graph.
    output_formatter : function
        The return parameters are passed through this function before being returned.
        It is required that all return values are able to transformed to pytensor.TensorVariable.
        Using this function, some return values can be discarded or transformed.
    name: str
        Name of the created pytensor Op, defaults to the name of the passed function.
    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """

    ### Construct the function to return that is compatible with pytensor but has the same signature as the jax function.
    def new_func(*args, **kwargs):
        func_signature = inspect.signature(jaxfunc)

        (
            input_treedef,
            input_arg_names,
            input_types,
            inputs_flat,
            other_args_dic,
        ) = _split_arguments(func_signature, args, kwargs, args_for_graph)

        ### Create internal function that accepts flattened inputs to use for pytensor.
        def conv_input_to_jax(func, flatten_output=True):
            def new_func(inputs_for_graph):
                inputs_for_graph = tree_unflatten(input_treedef, inputs_for_graph)
                inputs_for_graph = jax.tree_util.tree_map(
                    lambda x: jnp.array(x), inputs_for_graph
                )
                inputs_for_graph_dic = {
                    arg: val for arg, val in zip(input_arg_names, inputs_for_graph)
                }
                results = func(**inputs_for_graph_dic, **other_args_dic)
                if output_formatter is not None:
                    results = output_formatter(results)
                if not flatten_output:
                    return results
                else:
                    results, output_treedef_local = tree_flatten(results)

                    if len(results) > 1:
                        return tuple(
                            results
                        )  # Transform to tuple because jax makes a difference between tuple and list and not pytensor
                    else:
                        return results[0]

            return new_func

        jitted_sol_op_jax = jax.jit(
            conv_input_to_jax(jaxfunc),
            static_argnames=static_argnames,
        )

        # Convert static_argnames to static_argnums because make_jaxpr requires it.
        static_argnums = tuple(
            i
            for i, (k, param) in enumerate(func_signature.parameters.items())
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and k in static_argnames
        )

        ### Infer output shape and type from jax function, it works by passing pt.TensorType
        ### variables, as jax only needs type and shape information.
        input_types_w_neg_dim = []
        current_char = "a"
        for i, type in enumerate(input_types):
            if None in type.shape:
                # Replace None in shape with very negative number, as jax does not support
                # None in shape. By filtering afterwards for negative number, the dimensions
                # with unknown shape can be recovered again.

                # shape_new = tuple(
                #    [UnknDim() if dim is None else dim for dim in type.shape]
                # )
                str_shape = "("
                for dim in type.shape:
                    if dim is None:
                        str_shape += current_char
                        current_char = chr(ord(current_char) + 1)
                    else:
                        str_shape += str(dim)
                    str_shape += ", "
                str_shape = str_shape[:-2] + ")"
                shape_new = jax.experimental.export.shape_poly.symbolic_shape(str_shape)

                type_new = jax.core.ShapedArray(shape_new, type.dtype)
                # type_new = pt.TensorType(dtype=type.dtype, shape=shape_new)
                input_types_w_neg_dim.append(type_new)
                log.warning(
                    f"A dimension of input {i} is undefined: {type.shape}. The "
                    "transformation to pytensor might work, but it is experimental. "
                    "Open an issue if it does not work."
                )
            else:
                input_types_w_neg_dim.append(type)

        _, out_shape_jax = jax.make_jaxpr(
            conv_input_to_jax(jaxfunc, flatten_output=False),
            static_argnums=static_argnums,
            return_shape=True,
        )(input_types_w_neg_dim)

        out_shape_jax_flat, output_treedef = tree_flatten(out_shape_jax)
        output_types = [
            pt.TensorType(
                dtype=var.dtype,
                shape=tuple(
                    [None if jax._src.core.is_symbolic_dim(d) else d for d in var.shape]
                ),
            )
            for var in out_shape_jax_flat
        ]
        # if not current_char == "a":
        #    raise RuntimeError("Test")

        ### Create the Pytensor Op, the normal one and the vector-jacobian product (vjp)
        def vjp_sol_op_jax(args):
            y0 = args[:-len_gz]
            gz = args[-len_gz:]
            if len(gz) == 1:
                gz = gz[0]
            primals, vjp_fn = jax.vjp(local_op.perform_jax, *y0)
            gz = jax.tree_util.tree_map(
                lambda g, primal: jnp.broadcast_to(g, jnp.shape(primal)), gz, primals
            )
            if len(y0) == 1:
                return vjp_fn(gz)[0]
            else:
                return tuple(vjp_fn(gz))

        jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax)

        nonlocal name
        if name is None:
            name = jaxfunc.__name__

        SolOp, VJPSolOp = _return_pytensor_ops(name)

        local_op = SolOp(
            input_treedef,
            output_treedef,
            input_arg_names=input_arg_names,
            input_types=input_types,
            output_types=output_types,
            jitted_sol_op_jax=jitted_sol_op_jax,
            jitted_vjp_sol_op_jax=jitted_vjp_sol_op_jax,
            other_args=other_args_dic,
        )

        @jax_funcify.register(SolOp)
        def sol_op_jax_funcify(op, **kwargs):
            return local_op.perform_jax

        @jax_funcify.register(VJPSolOp)
        def vjp_sol_op_jax_funcify(op, **kwargs):
            return local_op.vjp_sol_op.perform_jax

        ### Evaluate the Pytensor Op and return unflattened results
        output_flat = local_op(*inputs_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        output = tree_unflatten(output_treedef, output_flat)
        len_gz = len(output_types)

        return output

    return new_func


def _split_arguments(func_signature, args, kwargs, args_for_graph):
    for key in kwargs.keys():
        if not key in func_signature.parameters and key in args_for_graph:
            raise RuntimeError(
                f"Keyword argument <{key}> not found in function signature. "
                f"**kwargs are not supported in the definition of the function,"
                f"because the order is not guaranteed."
            )
    arguments_bound = func_signature.bind(*args, **kwargs)
    arguments_bound.apply_defaults()

    # Check whether there exist an used **kwargs in the function signature
    for arg_name in arguments_bound.signature.parameters:
        if (
            arguments_bound.signature.parameters[arg_name]
            == inspect._ParameterKind.VAR_KEYWORD
        ):
            var_keyword = arg_name
    else:
        var_keyword = None
    arg_names = [
        key for key in arguments_bound.arguments.keys() if not key == var_keyword
    ]
    arg_names_from_kwargs = [key for key in arguments_bound.kwargs.keys()]

    if args_for_graph == "all":
        args_for_graph_from_args = arg_names
        args_for_graph_from_kwargs = arg_names_from_kwargs
    else:
        args_for_graph_from_args = []
        args_for_graph_from_kwargs = []
        for arg in args_for_graph:
            if arg in arg_names:
                args_for_graph_from_args.append(arg)
            elif arg in arg_names_from_kwargs:
                args_for_graph_from_kwargs.append(arg)
            else:
                raise ValueError(f"Argument {arg} not found in the function signature.")

    inputs_for_graph = [
        arguments_bound.arguments[arg] for arg in args_for_graph_from_args
    ]
    inputs_for_graph += [
        arguments_bound.kwargs[arg] for arg in args_for_graph_from_kwargs
    ]
    input_arg_names = args_for_graph_from_args + args_for_graph_from_kwargs
    other_args_dic = {
        arg: arguments_bound.arguments[arg]
        for arg in arg_names
        if not arg in input_arg_names
    }
    other_args_dic.update(
        **{
            arg: arguments_bound.kwargs[arg]
            for arg in arg_names_from_kwargs
            if not arg in input_arg_names
        }
    )

    # Convert our inputs to symbolic variables
    inputs_flat, input_treedef = tree_flatten(inputs_for_graph)
    inputs_flat = [pt.as_tensor_variable(inp) for inp in inputs_flat]

    input_types = [inp.type for inp in inputs_flat]

    return (
        input_treedef,
        input_arg_names,
        input_types,
        inputs_flat,
        other_args_dic,
    )


def _return_pytensor_ops(name):
    class SolOp(Op):
        def __init__(
            self,
            input_treedef,
            output_treeedef,
            input_arg_names,
            input_types,
            output_types,
            jitted_sol_op_jax,
            jitted_vjp_sol_op_jax,
            other_args,
        ):
            self.vjp_sol_op = None
            self.input_treedef = input_treedef
            self.output_treedef = output_treeedef
            self.input_arg_names = input_arg_names
            self.input_types = input_types
            self.output_types = output_types
            self.jitted_sol_op_jax = jitted_sol_op_jax
            self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax
            self.other_args = other_args

        def make_node(self, *inputs):
            self.num_inputs = len(inputs)

            # Define our output variables
            outputs = [pt.as_tensor_variable(type()) for type in self.output_types]
            self.num_outputs = len(outputs)

            self.vjp_sol_op = VJPSolOp(
                self.input_treedef,
                self.input_types,
                self.jitted_vjp_sol_op_jax,
                self.other_args,
            )

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            results = self.jitted_sol_op_jax(inputs)
            if self.num_outputs > 1:
                for i in range(self.num_outputs):
                    outputs[i][0] = np.array(results[i], self.output_types[i].dtype)
            else:
                outputs[0][0] = np.array(results, self.output_types[0].dtype)

        def perform_jax(self, *inputs):
            results = self.jitted_sol_op_jax(inputs)
            return results

        def grad(self, inputs, output_gradients):
            # If a output is not used, it is disconnected and doesn't have a gradient.
            # Set gradient here to zero for those outputs.
            # raise NotImplementedError()
            for i in range(self.num_outputs):
                if isinstance(output_gradients[i].type, DisconnectedType):
                    if not None in self.output_types[i].shape:
                        output_gradients[i] = pt.zeros(
                            self.output_types[i].shape, self.output_types[i].dtype
                        )
                    else:
                        output_gradients[i] = pt.zeros((), self.output_types[i].dtype)
            result = self.vjp_sol_op(inputs, output_gradients)

            if self.num_inputs > 1:
                return result
            else:
                return (result,)  # Pytensor requires a tuple here

    # vector-jacobian product Op
    class VJPSolOp(Op):
        def __init__(
            self, input_treedef, input_types, jitted_vjp_sol_op_jax, other_args
        ):
            self.input_treedef = input_treedef
            self.input_types = input_types
            self.jitted_vjp_sol_op_jax = jitted_vjp_sol_op_jax
            self.other_args = other_args

        def make_node(self, y0, gz):
            y0 = [
                pt.as_tensor_variable(
                    _y,
                ).astype(self.input_types[i].dtype)
                for i, _y in enumerate(y0)
            ]
            gz_not_disconntected = [
                pt.as_tensor_variable(_gz)
                for _gz in gz
                if not isinstance(_gz.type, DisconnectedType)
            ]
            # self.num_gz_not_disconnected = len(gz_not_disconntected)

            outputs = [in_type() for in_type in self.input_types]
            self.num_outputs = len(outputs)
            return Apply(self, y0 + gz_not_disconntected, outputs)

        def perform(self, node, inputs, outputs):
            # inputs = tree_unflatten(self.full_input_treedef_def, inputs)
            # y0 = inputs[:-self.num_gz]
            # gz = inputs[-self.num_gz:]
            results = self.jitted_vjp_sol_op_jax(tuple(inputs))
            if len(self.input_types) > 1:
                for i, result in enumerate(results):
                    outputs[i][0] = np.array(results[i], self.input_types[i].dtype)
            else:
                outputs[0][0] = np.array(results, self.input_types[0].dtype)

        def perform_jax(self, *inputs):
            # inputs = tree_unflatten(self.full_input_treedef_def, inputs)

            results = self.jitted_vjp_sol_op_jax(tuple(inputs))
            if self.num_outputs == 1:
                if isinstance(results, Sequence):
                    return results[0]
                else:
                    return results
            else:
                return tuple(results)

    SolOp.__name__ = name
    SolOp.__qualname__ = ".".join(SolOp.__qualname__.split(".")[:-1] + [name])

    VJPSolOp.__name__ = "VJP_" + name
    VJPSolOp.__qualname__ = ".".join(
        VJPSolOp.__qualname__.split(".")[:-1] + ["VJP_" + name]
    )

    return SolOp, VJPSolOp
