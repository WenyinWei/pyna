from typing import Any
import sympy

class Flow:
    @property
    def arg_dim(self):
        raise NotImplementedError()
    @property
    def value_dim(self):
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()

class FlowSympy(Flow):
    def __init__(self, xi_syms:list, diff_xi_exprs:list, param_dict:dict = None):
        self._xi_syms = xi_syms
        self._diff_xi_exprs = diff_xi_exprs
        if param_dict is None:
            self._param_dict = dict()
        else:
            self._param_dict = param_dict

    @property
    def xi_syms(self):
        return self._xi_syms
    @property
    def arg_dim(self) -> int:
        return len(self.xi_syms)
    @property
    def diff_xi_exprs(self):
        return self._diff_xi_exprs
    @property
    def value_dim(self) -> int:
        return len(self.diff_xi_exprs)

    @property
    def free_symbols(self) -> sympy.sets.sets.FiniteSet:
        from functools import reduce
        from operator import or_
        return reduce(or_, [sympy.FiniteSet(*func.free_symbols) for func in self.diff_xi_exprs])
    @property
    def free_params(self) -> sympy.sets.sets.FiniteSet:
        return self.free_symbols - sympy.FiniteSet(*self.xi_syms)
    @property
    def param_dict(self):
        return self._param_dict
    @param_dict.setter
    def param_dict(self, param_dict_:dict): # you can update the parameter dict, but must as a whole.
        if not isinstance(param_dict_, dict):
            raise ValueError("The param_dict arg must be a python dict object.")
        for key in param_dict_.keys():
            if not key in self.free_symbols:
                raise ValueError("Your input `param_dict` contains some weird symbol(s) which do(es)n't appear in the function sympy expressions.")
        self._param_dict = param_dict_

    def param_dict_cover_free_symbols(self) -> bool:
        if (self.free_params - sympy.FiniteSet( *self.param_dict.keys() )).is_empty:
            return True
        else:
            return False


    def diff_xi_lambdas(self, lambda_type:str = "numpy"):
        from .sysutil import check_lambdify_package_available
        check_lambdify_package_available(lambda_type)
        if not self.param_dict_cover_free_symbols():
            raise ValueError("Missing param value, check if every free symbol has been filled values by setting param_dict.")

        if lambda_type == "numpy":
            lambda_list = [sympy.lambdify(self.xi_syms, func.subs(self.param_dict)) for func in self.diff_xi_exprs]
        else:
            raise NotImplementedError("Not yet prepared for other lambda type than 'numpy'.")
        return lambda_list

    def __call__(self, xi_arrays:list, lambda_type:str = "numpy"):
        return (lam(*xi_arrays) for lam in self.diff_xi_lambdas(lambda_type=lambda_type))


class FlowCallable(Flow):
    def __init__(self, diff_xi_funcs:list) -> None:
        super().__init__()
        self._diff_xi_funcs = diff_xi_funcs

    @property
    def arg_dim(self):
        raise NotImplementedError()
    @property
    def diff_xi_funcs(self):
        return self._diff_xi_funcs
    @property
    def value_dim(self):
        raise len(self._diff_xi_funcs)


    def diff_xi_lambdas(self, lambda_type:str = None):
        if lambda_type is not None:
            raise ValueError("The Flow is already FlowCallable, clients can no longer specify which way to lambdify the functions.  ")
        return self.diff_xi_funcs

    def __call__(self, xi_arrays: list):
        return (lam(*xi_arrays) for lam in self.diff_xi_lambdas)
