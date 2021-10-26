from typing import Any
import sympy

from pyna.withparam import WithParam

class Map:
    @property
    def arg_dim(self):
        raise NotImplementedError()
    @property
    def value_dim(self):
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()

class MapSympy(Map, WithParam):
    def __init__(self, xi_syms:list, next_xi_exprs:list, param_dict:dict = None):
        self._xi_syms = xi_syms
        self._next_xi_exprs = next_xi_exprs
        WithParam.__init__(self, param_dict)

    @property
    def xi_syms(self):
        return self._xi_syms
    @property
    def arg_dim(self) -> int:
        return len(self.xi_syms)
    @property
    def next_xi_exprs(self):
        return self._next_xi_exprs
    @property
    def value_dim(self) -> int:
        return len(self.next_xi_exprs)

    @property
    def free_symbols(self) -> sympy.sets.sets.FiniteSet:
        from functools import reduce
        from operator import or_
        return reduce(or_, [sympy.FiniteSet(*func.free_symbols) for func in self.next_xi_exprs])
    @property
    def free_params(self) -> sympy.sets.sets.FiniteSet:
        return self.free_symbols - sympy.FiniteSet(*self.xi_syms)

    def next_xi_lambdas(self, lambda_type:str = "numpy"):
        from .sysutil import check_lambdify_package_available
        check_lambdify_package_available(lambda_type)
        if not self.param_dict_cover_free_symbols():
            raise ValueError("Missing param value, check if every free symbol has been filled values by setting param_dict.")

        if lambda_type == "numpy":
            lambda_list = [sympy.lambdify(self.xi_syms, func.subs(self.param_dict)) for func in self.next_xi_exprs]
        else:
            raise NotImplementedError("Not yet prepared for other lambda type than 'numpy'.")
        return lambda_list

    def __call__(self, xi_arrays:list, lambda_type:str = "numpy"):
        return (lam(*xi_arrays) for lam in self.next_xi_lambdas(lambda_type=lambda_type))

    def __or__(self, other):
        """pipeline | operator

        Args:
            other (Map): The other Map to be pipelined into.

        Returns:
            Map: The composite of `self` and `other` maps.

        Note:
            The pipeline operator is very dedicated (and fragile for developers who have little knowledge about Pyhton scope rules) in order to achieve dynamic polymorphism. Notice that we delay the definition of MapBuilder until the definitions of Map, MapSameDim, Map1D and Map2D, which fully utilize the power of polymorphism of Python. Please refer to [StackOverflow: Declaration functions in python after call](https://stackoverflow.com/questions/17953219/declaration-functions-in-python-after-call) for tutorial on how this works.
        """
        if isinstance(other, MapSympy):
            return MapSympyComposite(self, other)
        else:
            raise NotImplementedError()


class MapSympyAdd(MapSympy): # To support +/- operator on Map
    pass
class MapSympyMul(MapSympy): # Support scalar mul
    pass
class MapSympyComposite(MapSympy):
    def __init__(self, first_map: MapSympy, second_map: MapSympy):
        self._first_map = first_map
        self._second_map = second_map

    @property
    def xi_syms(self):
        return self._first_map.xi_syms
    @property
    def arg_dim(self) -> int:
        return self._first_map.arg_dim
    @property
    def next_xi_exprs(self):
        sym_subs_dict = {key: self._first_map.next_xi_exprs[i] for i, key in enumerate(self._second_map.xi_syms)}
        return [func.subs(sym_subs_dict) for func in self._second_map.next_xi_exprs]
    @property
    def value_dim(self) -> int:
        return self._second_map.value_dim

    @property
    def param_dict(self):
        return self._first_map.param_dict | self._second_map._param_dict
    @param_dict.setter
    def param_dict(self, param_dict_:dict):
        for key in param_dict_.keys():
            if key in self._first_map.keys():
                self._first_map._param_dict[key, param_dict_[key]]
            if key in self._second_map.keys():
                self._second_map._param_dict[key, param_dict_[key]]
    def __call__(self, xi_arrays: list, lambda_type: str = "numpy"):
        return self._second_map(
                    self._first_map(xi_arrays, lambda_type=lambda_type), 
                lambda_type=lambda_type )

class MapCallable(Map):
    def __init__(self, next_xi_funcs:list) -> None:
        super().__init__()
        self._next_xi_funcs = next_xi_funcs

    @property
    def arg_dim(self):
        raise NotImplementedError()
    @property
    def next_xi_funcs(self):
        return self._next_xi_funcs
    @property
    def value_dim(self):
        raise len(self.next_xi_funcs)


    def next_xi_lambdas(self, lambda_type:str = None):
        if lambda_type is not None:
            raise ValueError("The Map is already MapCallable, clients can no longer specify which way to lambdify the functions. ")
        return self.next_xi_funcs

    def __call__(self, xi_arrays: list):
        return (lam(*xi_arrays) for lam in self.next_xi_lambdas)

class MapUniCallable(MapCallable):
    def __init__(self, next_xi_func: callable) -> None:
        super().__init__()
        self._next_xi_func = next_xi_func

    @property
    def arg_dim(self):
        raise NotImplementedError()
    @property
    def next_xi_funcs(self):
        raise NotImplementedError()
    @property
    def value_dim(self):
        raise len(self.next_xi_funcs)


    def next_xi_lambdas(self, lambda_type:str = None):
        if lambda_type is not None:
            raise ValueError("The Map is already MapCallable, clients can no longer specify which way to lambdify the functions. ")
        raise NotImplementedError()

    def __call__(self, xi_arrays: list):
        return self._next_xi_func(*xi_arrays)

class MapCallableComposite(MapCallable):
    def __init__(self, first_map: MapCallable, second_map: MapCallable):
        self._first_map = first_map
        self._second_map = second_map

    @property
    def arg_dim(self) -> int:
        return self._first_map.arg_dim
    @property
    def next_xi_funcs(self):
        raise NotImplementedError()
    @property
    def value_dim(self) -> int:
        return self._second_map.value_dim
    def __call__(self, xi_arrays: list, lambda_type: str = "numpy"):
        return self._second_map( self._first_map(xi_arrays) )