
import sympy


class Map:
    def __init__(self, xi_syms:list, next_xi_funcs:list):
        self._xi_syms = xi_syms

        if isinstance( next_xi_funcs[0], sympy.Basic ):
            self._func_type = "sympy_expr"
            for func in next_xi_funcs:
                if not isinstance( func, sympy.Basic ):
                    raise ValueError("Make sure all functions are of the same root type, i.e., for sympy that is sympy.Basic.")
        elif callable( next_xi_funcs[0] ):
            self._func_type = "whatever_callable"
        else:
            raise ValueError("The input next_xi_funcs is not callable.")
        
        self._next_xi_funcs = next_xi_funcs

        self._lambda_type = "numpy"
        self._param_dict = dict()

    @property
    def xi_syms(self):
        return self._xi_syms
    @property
    def arg_dim(self):
        return len(self.xi_syms)
    @property
    def next_xi_funcs(self):
        return self._next_xi_funcs
    @property
    def value_dim(self):
        return len(self.next_xi_funcs)
    @property
    def func_type(self):
        return self._func_type

    @property
    def free_symbols(self) -> sympy.sets.sets.Set:
        from functools import reduce
        from operator import or_
        return reduce(or_, [func.free_symbols for func in self._next_xi_funcs])
    @property
    def param_dict(self):
        return self._param_dict
    @param_dict.setter
    def param_dict(self, param_dict_value:dict):
        if not isinstance(param_dict_value, dict):
            raise ValueError("The param_dict arg must be a python dict object.")
        for key in param_dict_value.keys():
            if not key in self.free_symbols:
                raise ValueError("Your input `param_dict` contains some weird symbol(s) which do(es)n't appear in the function sympy expressions.")
        self._param_dict = param_dict_value
    @property
    def param_dict_cover_free_symbols(self) -> bool:
        if (self.free_symbols - sympy.FiniteSet( self.param_dict.keys() )).is_empty:
            return True
        else:
            return False

    def next_xi_lambdas(self, lambda_type:str = None):
        if lambda_type is None:
            if self._lambda_type is None:
                self._lambda_type = "numpy"

        if self._func_type == "sympy_expr":
            if self._lambda_type == "numpy":
                lambda_list = [sympy.lambdify(self.xi_syms, func.subs(self.param_dict)) for func in self._next_xi_funcs]
            else:
                raise NotImplementedError("Not yet prepared for other lambda type than 'numpy'.")
            return lambda_list
        else:
            raise NotImplementedError("Not yet prepared for other function type than 'sympy_expr'.")

    def __call__(self, xi_arrays:list, lambda_type:str = None):
        return (lam(*xi_arrays) for lam in self.next_xi_lambdas(lambda_type=lambda_type))

    def __or__(self, other):
        sym_subs_dict = {key: self.next_xi_funcs[i] for i, key in enumerate(other.xi_syms)}
        return MapBuilder(self.xi_syms, [func.subs(sym_subs_dict) for func in other.next_xi_funcs])

class MapSameDim(Map):
    def __init__(self, xi_syms: list, next_xi_funcs: list):
        if len(xi_syms) != len(next_xi_funcs):
            raise ValueError("For MapSameDim, the arg and value dimensions shall be the same.")
        super().__init__(xi_syms, next_xi_funcs)
    def __or__(self, other:Map):
        return super().__or__(other) 
    def inv(self):
        raise NotImplementedError()
        
class Map1D(MapSameDim):
    def __init__(self, xi_syms:list, next_xi_funcs:list):
        if len(xi_syms) != 1 or len(next_xi_funcs) != 1:
            raise ValueError("For Map1D, a one-dimensional dynamic system should be input, check your input .")
        super().__init__(xi_syms, next_xi_funcs)
    def __or__(self, other:Map):
        return super().__or__(other) 
class Map2D(MapSameDim):
    def __init__(self, xi_syms:list, next_xi_funcs:list):
        if len(xi_syms) != 2 or len(next_xi_funcs) != 2:
            raise ValueError("For Map2D, a two-dimensional dynamic system should be input, check your input .")
        super().__init__(xi_syms, next_xi_funcs)
    def __or__(self, other:Map):
        return super().__or__(other) 

def MapBuilder(xi_syms:list, next_xi_funcs:list):
    if len(xi_syms) == len(next_xi_funcs) == 1:
        return Map1D(xi_syms, next_xi_funcs)
    elif len(xi_syms) == len(next_xi_funcs) == 2:
        return Map2D(xi_syms, next_xi_funcs)
    elif len(xi_syms) == len(next_xi_funcs):
        return MapSameDim(xi_syms, next_xi_funcs)
    else:
        return Map(xi_syms, next_xi_funcs)