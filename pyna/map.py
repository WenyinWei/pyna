from typing import Any
import sympy

class Map:
    @property
    def arg_dim(self):
        raise NotImplementedError()
    @property
    def value_dim(self):
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()

class MapSympy(Map):
    def __init__(self, xi_syms:list, next_xi_exprs:list, param_dict:dict = dict()):
        self._xi_syms = xi_syms
        self._next_xi_exprs = next_xi_exprs
        self._param_dict = param_dict

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
    def free_symbols(self) -> sympy.sets.sets.Set:
        from functools import reduce
        from operator import or_
        return reduce(or_, [func.free_symbols for func in self.next_xi_exprs])
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
    @property
    def param_dict_cover_free_symbols(self) -> bool:
        if (self.free_symbols - sympy.FiniteSet( self.param_dict.keys() )).is_empty:
            return True
        else:
            return False

    @staticmethod
    def check_lambda_package_available(lambda_type_value:str):
        from .sysutil import if_package_installed
        if lambda_type_value == "numpy":
            if not if_package_installed("numpy"):
                raise ImportError("The lambdifying requires numpy package.")
        elif lambda_type_value == "cupy":
            if not if_package_installed("cupy"):
                raise ImportError("The lambdifying requires cupy package.")
        else:
            raise NotImplementedError()
    def next_xi_lambdas(self, lambda_type:str = "numpy"):
        MapSympy.check_lambda_package_available(lambda_type)

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


    def __call__(self, xi_arrays: list):
        return (lam(*xi_arrays) for lam in self.next_xi_lambdas)

        
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