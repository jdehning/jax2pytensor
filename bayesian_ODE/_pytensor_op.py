"""Convert a jax function to a pytensor Op."""

import inspect
from collections.abc import Sequence

import jax
import numpy as np
import pytensor.tensor as pt
import pytensor.scalar as ps
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from functools import partial


def create_and_register_jax(
    jax_func,
    output_shape_func=None,
    name=None,
    args_for_graph="all",
    static_argnums=(),
):
    """Return a pytensor from a jax jittable function.

    It requires to define the output types of the returned values as pytensor types. A
    unique name should also be passed in case the name of the jax_func is identical to
    some other node. The design of this function is based on
    https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick-examples/

    Parameters
    ----------
    jax_func : jax jittable function
        function for which the node is created, can return multiple tensors as a tuple.
    args_for_graph :
        If "all", all arguments except arguments passed via **kwargs are used for the graph.
    output_shape_func : function
        Function that returns the shape of the output. If None, the shape is expected to
        be the same as the shape of the args_for_graph arguments. If not None, the function
        should return a tuple of shapes, it will receive as input the shapes of the args_for_graph
        as tuples. Shapes are defined as tuples of integers or None.
    name: str
        Name of the created pytensor Op, defaults to the name of the passed function.
        Should be unique so that jax_juncify won't ovewrite another when registering it
    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """

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
            results = self.jitted_sol_op_jax(inputs, **self.other_args)
            if self.num_outputs > 1:
                for i in range(self.num_outputs):
                    outputs[i][0] = np.array(results[i], self.output_types[i].dtype)
            else:
                outputs[0][0] = np.array(results, self.output_types[0].dtype)

        def perform_jax(self, *inputs):
            results = self.jitted_sol_op_jax(inputs, **self.other_args)
            return results

        def grad(self, inputs, output_gradients):
            # If a output is not used, it is disconnected and doesn't have a gradient.
            # Set gradient here to zero for those outputs.
            # raise NotImplementedError()
            for i in range(self.num_outputs):
                if isinstance(output_gradients[i].type, DisconnectedType):
                    #        # output_gradients[i] = pt.zeros(otype.shape, otype.dtype)
                    #        output_gradients[i] = pt.zeros(
                    #            self.output_types[i].shape, self.output_types[i].dtype
                    #        )
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
            print(len(outputs))
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

    def new_func(*args, **kwargs):
        func_signature = inspect.signature(jax_func)

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

        # all_inputs = arguments.args

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
                    raise ValueError(
                        f"Argument {arg} not found in the function signature."
                    )

        inputs_for_graph = [
            arguments_bound.arguments[arg] for arg in args_for_graph_from_args
        ]
        inputs_for_graph += [
            arguments_bound.kwargs[arg] for arg in args_for_graph_from_kwargs
        ]
        all_args_for_graph = args_for_graph_from_args + args_for_graph_from_kwargs
        other_args_dic = {
            arg: arguments_bound.arguments[arg]
            for arg in arg_names
            if not arg in all_args_for_graph
        }
        other_args_dic.update(
            **{
                arg: arguments_bound.kwargs[arg]
                for arg in arg_names_from_kwargs
                if not arg in all_args_for_graph
            }
        )

        # Convert our inputs to symbolic variables
        all_inputs_flat, input_treedef = tree_flatten(inputs_for_graph)
        all_inputs_flat = [pt.as_tensor_variable(inp) for inp in all_inputs_flat]

        input_types = [inp.type for inp in all_inputs_flat]

        output_dtype = ps.upcast(*[inp_type.dtype for inp_type in input_types])

        # len_gz = len(output_types)

        nonlocal output_shape_func
        if output_shape_func is None:
            output_shape_func = lambda **shape_dic: tuple(shape_dic.values())

        input_shapes_list = tree_unflatten(
            input_treedef, [inp_type.shape for inp_type in input_types]
        )
        input_shapes_dic = {
            arg: shape for arg, shape in zip(all_args_for_graph, input_shapes_list)
        }

        print(output_shape_func(**input_shapes_dic))

        # For flattening the output shapes, we need to redefine what is a leaf, so that
        # the shape tuples don't get also flattened.
        is_leaf = lambda x: isinstance(x, Sequence) and (
            len(x) == 0 or x[0] is None or isinstance(x[0], int)
        )
        output_shape = output_shape_func(**input_shapes_dic)
        # if isinstance(output_shape, Sequence):
        #    output_shape = tuple(output_shape)

        output_shapes_flat, output_treedef = tree_flatten(output_shape, is_leaf=is_leaf)
        # output_shapes_flat = tuple(output_shapes_flat)

        if len(output_shapes_flat) == 0 or not isinstance(
            output_shapes_flat[0], Sequence
        ):
            output_shapes_flat = (output_shapes_flat,)
        print(output_shapes_flat)

        output_types = [
            pt.type.TensorType(dtype=output_dtype, shape=shape)
            for shape in output_shapes_flat
        ]
        len_gz = len(output_types)

        def conv_input_to_jax(func):
            def new_func(inputs_for_graph, **other_args_kwargs):
                inputs_for_graph = tree_unflatten(input_treedef, inputs_for_graph)
                inputs_for_graph = jax.tree_util.tree_map(
                    lambda x: jnp.array(x), inputs_for_graph
                )
                inputs_for_graph_dic = {
                    arg: val for arg, val in zip(all_args_for_graph, inputs_for_graph)
                }
                results = func(**inputs_for_graph_dic, **other_args_dic)
                results, output_treedef_local = tree_flatten(results)
                if output_treedef_local != output_treedef:
                    raise ValueError(
                        f"The output of the jax function {output_treedef_local} does not match the tree definition"
                        f"inferred from the output_shape_func: {self.output_treedef}. Please check the output_shape_func."
                    )
                # remove_size_one_dim = lambda x: x[0] if jnp.shape(x) == (1,) else x
                # results = jax.tree_map(remove_size_one_dim, results)
                if len_gz > 1:
                    return tuple(
                        results
                    )  # Transform to tuple because jax makes a difference between tuple and list
                else:
                    return results[0]

            return new_func

        jitted_sol_op_jax = jax.jit(
            conv_input_to_jax(jax_func), static_argnames=tuple(other_args_dic.keys())
        )

        def vjp_sol_op_jax(args):
            y0 = args[:-len_gz]
            gz = args[-len_gz:]
            # for i, gz_i in enumerate(gz):
            #    if jnp.shape(gz_i) == ():
            #        gz_i = jnp.where(jnp.isnan(gz_i), jnp.zeros_like(y0[i]), gz_i)

            # remove_size_one_dim = lambda x: x[0] if jnp.shape(x) == (1,) else x
            # y0 = jax.tree_map(remove_size_one_dim, y0)
            # gz = jax.tree_map(remove_size_one_dim, gz)

            # gz = tree_unflatten(output_treedef, gz_flat)
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
            name = jax_func.__name__
        SolOp.__name__ = name
        SolOp.__qualname__ = ".".join(SolOp.__qualname__.split(".")[:-1] + [name])

        VJPSolOp.__name__ = "VJP_" + name
        VJPSolOp.__qualname__ = ".".join(
            VJPSolOp.__qualname__.split(".")[:-1] + ["VJP_" + name]
        )

        local_op = SolOp(
            input_treedef,
            output_treedef,
            input_arg_names=all_args_for_graph,
            input_types=input_types,
            output_types=output_types,
            jitted_sol_op_jax=jitted_sol_op_jax,
            jitted_vjp_sol_op_jax=jitted_vjp_sol_op_jax,
            other_args=other_args_dic,
        )

        output_flat = local_op(*all_inputs_flat)
        if not isinstance(output_flat, Sequence):
            output_flat = [output_flat]  # tree_unflatten expects a sequence.
        output = tree_unflatten(output_treedef, output_flat)

        @jax_funcify.register(SolOp)
        def sol_op_jax_funcify(op, **kwargs):
            return local_op.perform_jax

        @jax_funcify.register(VJPSolOp)
        def vjp_sol_op_jax_funcify(op, **kwargs):
            return local_op.vjp_sol_op.perform_jax

        return output

    return new_func
