import logging

import diffrax
from pytensor.tensor.type import TensorType
import jax

from ._pytensor_op import create_and_register_jax

logger = logging.getLogger(__name__)


class ODEIntegrator:
    """Creates an integrator for compartmental models.

    The integrator is a function that
    takes as input a function that returns the derivatives of the variables of the system
    of differential equations, and returns a function that solves the system of
    differential equations. For the initilization of the integrator object, the timesteps
    of the solver and the output have to be specified.

    """

    def __init__(
        self,
        ts_out,
        ts_solver=None,
        ts_arg=None,
        interp="cubic",
        solver=diffrax.Tsit5(),
        t_0=None,
        t_1=None,
        **kwargs,
    ):
        """Initialize the ODEIntegrator class.

        Parameters
        ----------
        ts_out : array-like
            The timesteps at which the output is returned.
        ts_solver : array-like or None
            The timesteps at which the solver will be called. If None, it is set to ts_out.
        ts_arg : array-like or None
            The timesteps at which the time-dependent argument of the system of differential
            equations are given. If None, it is set to ts_solver.
        interp : str
            The interpolation method used to interpolate_pytensor the time-dependent argument of the
            system of differential equations. Can be "cubic" or "linear".
        solver : :class:`diffrax.AbstractStepSizeController`
            The solver used to integrate the system of differential equations. Default is
            diffrax.Tsit5(), a 5th order Runge-Kutta method.
        t_0 : float or None
            The initial time of the integration. If None, it is set to ts_solve[0].
        t_1 : float or None
            The final time of the integration. If None, it is set to ts_solve[-1].
        **kwargs
            Arguments passed to the solver, see :func:`diffrax.diffeqsolve` for more details.
        """
        self.ts_out = ts_out
        if ts_solver is None:
            self.ts_solver = self.ts_out
        else:
            self.ts_solver = ts_solver
        if t_0 is None:
            self.t_0 = float(self.ts_solver[0])
        else:
            self.t_0 = t_0
        if self.t_0 > self.ts_out[0]:
            raise ValueError("t_0 should be smaller than the first element of ts_out")
        if t_1 is None:
            self.t_1 = float(self.ts_solver[-1])
        else:
            self.t_1 = t_1
        if self.t_1 < self.ts_out[-1]:
            raise ValueError("t_1 should be larger than the last element of ts_out")
        self.ts_arg = ts_arg
        self.interp = interp
        self.solver = solver
        self.kwargs_solver = kwargs

    def get_func(self, ODE, list_keys_to_return=None):
        """
        Return a function that solves the system of differential equations.

        Parameters
        ----------
        ODE : function(t, y, args)
            A function that returns the derivatives of the variables of the system of
            differential equations. The function has to take as input the time `t`, the
            variables `y` and the arguments `args=(arg_t, constant_args)` of the system
            of differential equations.
            `t` is a float, `y` is a list or dict of floats or ndarrays, or in general, a
            pytree, see :mod:`jax.tree_util` for more details. The return value of the function
            has to be a pytree/list/dict with the same structure as `y`.
        list_keys_to_return : list of str or None, default is None
            The keys of the variables of the system of differential equations that are
            returned by the integrator. If set, the integrator returns a list of the
            variables of the system of differential equations in the order of the keys.
            If `None`, the output is returned as is.

        Returns
        -------
        integrator : function(y0, arg_t=None, constant_args=None)
            A function that solves the system of differential equations and returns the
            output at the specified timesteps. The function takes as input `y0` the initial
            values of the variables of the system of differential equations, the
            time-dependent argument of the system of differential equations `arg_t`, and
            the constant arguments `constant_args` of the system of differential equations. `t`, `y0` and
            `(arg_t, constant_args)` are passed to the ODE function as its three arguments.
            If `arg_t` is `None`, only `constant_args` are passed to the ODE function and
            vice versa, without being in a tuple.

        """

        def integrator(y0, arg_t=None, constant_args=None):
            if arg_t is not None:
                if not callable(arg_t):
                    if self.ts_arg is None:
                        raise RuntimeError("Specify ts_arg to use a non-callable arg_t")
                    arg_t_func = interpolation_func(
                        ts=self.ts_arg, x=arg_t, method=self.interp
                    ).evaluate
                else:
                    logger.warning(
                        "arg_t is callable, but ts_arg is not None. ts_arg"
                        " won't be used."
                    )
                    arg_t_func = arg_t

            if arg_t is None and self.ts_arg is not None:
                logger.warning(
                    "You did specify ts_arg, but arg_t is None. Did you mean to do this?"
                )
            term = diffrax.ODETerm(ODE)

            if arg_t is None:
                args = constant_args
            elif constant_args is None:
                args = arg_t_func
            else:
                args = (
                    arg_t_func,
                    constant_args,
                )
            saveat = diffrax.SaveAt(ts=self.ts_out)  # jnp.array?

            stepsize_controller = (
                diffrax.StepTo(ts=self.ts_solver)  # jnp.array?
                if not "stepsize_controller" in self.kwargs_solver
                else self.kwargs_solver["stepsize_controller"]
            )

            dt0 = (
                None
                if isinstance(stepsize_controller, diffrax.StepTo)
                else self.ts_solver[1] - self.ts_solver[0]
            )

            sol = diffrax.diffeqsolve(
                term,
                self.solver,
                self.t_0,
                self.t_1,
                dt0=dt0,
                stepsize_controller=stepsize_controller,
                y0=y0,
                args=args,
                saveat=saveat,
                **self.kwargs_solver,
                # adjoint=diffrax.BacksolveAdjoint(),
            )
            if list_keys_to_return is None:
                return sol.ys
            else:
                return tuple([sol.ys[key] for key in list_keys_to_return])

        return integrator

    def get_op(
        self,
        ODE,
        return_shapes=((),),
        list_keys_to_return=None,
        name=None,
    ):
        """Return a pytensor operator that solves the system of differential equations.

        Same as get_func, but returns a pytensor operator that can be used in a pymc model.
        Beware that for this operator the output of the integration of the ODE can only
        be a single or a list variables. If the output is a dict, set list_keys_to_return
        to specify the keys of the variables that are returned by the integrator. These
        return values aren't allowed to be further nested.

        Parameters
        ----------
        ODE : function(t, y, args)
            A function that returns the derivatives of the variables of the system of
            differential equations. The function has to take as input the time `t`, the
            variables `y` and the arguments `args=(arg_t, constant_args)` of the system
            of differential equations.
            `t` is a float, `y` is a list or dict of floats or ndarrays, or in general, a
            pytree, see :mod:`jax.tree_util` for more details. The return value of the function
            has to be a pytree/list/dict with the same structure as `y`.
        return_shapes : tuple of tuples, default is ((),)
            The shapes (except the time dimension) of the variables of the system of
            differential equations that are returned by the integrator. If
            `list_keys_to_return` is `None`, the shapes have to be given in the same order
            as the variables are returned by the integrator. If `list_keys_to_return` is
            not `None`, the shapes have to be given in the same order as the keys in
            `list_keys_to_return`. The default `((),)` means a single variable with only a
            time dimension is returned.
        list_keys_to_return : list of str or None, default is None
            The keys of the variables of the system of differential equations that will
            be chosen to be returned by the integrator. Necessary if the ODE returns a
            `dict`, as :mod:`pytensor` only accepts single outputs or a list of outputs.
            If `None`, the output is returned as is.
        name :
            The name under which the operator is registered in pymc.

        Returns
        -------
        pytensor_op : :class:`pytensor.graph.op.Op`
            A :mod:`pytensor` operator that can be used in a :class:`pymc.Model`.

        """
        integrator = self.get_func(ODE, list_keys_to_return=list_keys_to_return)

        pytensor_op = create_and_register_jax(
            integrator,
            output_types=[
                TensorType(
                    dtype="float64", shape=tuple([len(self.ts_out)] + list(shape))
                )
                for shape in return_shapes
            ],
            name=name,
        )
        return pytensor_op


def interpolation_func(ts, x, method="cubic"):
    """
    Return a diffrax-interpolation function that can be used to interpolate_pytensor the time-dependent variable.

    Parameters
    ----------
    ts : array-like
        The timesteps at which the time-dependent variable is given.
    x : array-like
        The time-dependent variable.
    method
        The interpolation method used. Can be "cubic" or "linear".

    Returns
    -------
    interp : :class:`diffrax.CubicInterpolation` or :class:`diffrax.LinearInterpolation`
        The interpolation function. Call `interp.evaluate(t)` to evaluate the interpolated
        variable at time `t`. t can be a float or an array-like.

    """
    # ts = jnp.array(ts)
    if method == "cubic":
        coeffs = diffrax.backward_hermite_coefficients(ts, x)
        interp = diffrax.CubicInterpolation(ts, coeffs)
    elif method == "linear":
        interp = diffrax.LinearInterpolation(ts, x)
    else:
        raise RuntimeError(
            f'Interpoletion method {method} not known, possibilities are "cubic" or "linear"'
        )
    return interp


def interpolate_pytensor(
    ts_in, ts_out, y, method="cubic", ret_gradients=False, name=None
):
    """
    Interpolate the time-dependent variable `y` at the timesteps `ts_out`.

    Parameters
    ----------
    ts_in : array-like
        The timesteps at which the time-dependent variable is given.
    ts_out : array-like
        The timesteps at which the time-dependent variable should be interpolated.
    y : array-like
        The time-dependent variable.
    method : str
        The interpolation method used. Can be "cubic" or "linear".
    ret_gradients : bool
        If True, the gradients of the interpolated variable are returned. Default is
        False.

    Returns
    -------
    y_interp : array-like
        The interpolated variable at the timesteps `ts_out`.

    """

    def interpolator(ts_out, y, ts_in=ts_in):
        interp = interpolation_func(ts_in, y, method)
        if ret_gradients:
            return jax.vmap(interp.derivative, 0, 0)(ts_out)
        else:
            return jax.vmap(interp.evaluate, 0, 0)(ts_out)

    interpolator_op = create_and_register_jax(
        interpolator,
        output_types=[
            TensorType(dtype="float64", shape=(len(ts_out),)),
        ],
        name=name,
    )

    return interpolator_op(ts_out, y)
