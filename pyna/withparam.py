import sympy

class WithParam:
    def __init__(self, param_dict: dict = None) -> None:
        if param_dict is None:
            self.param_dict = dict()
        else:
            self.param_dict = param_dict

    # NOTE: pure virtual method
    @property
    def free_symbols(self) -> sympy.sets.sets.FiniteSet: 
        # Reference Implementation
        # from functools import reduce
        # from operator import or_
        # return reduce(or_, [sympy.FiniteSet(*func.free_symbols) for func in self.diff_xi_exprs])
        raise NotImplementedError()

    # NOTE: pure virtual method
    @property
    def free_params(self) -> sympy.sets.sets.FiniteSet: 
        # Reference Implementation
        # return self.free_symbols - sympy.FiniteSet(*self.xi_syms)
        raise NotImplementedError()

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

from sympy.tensor.array import ImmutableDenseNDimArray
class ImmutableDenseNDimArrayWithParam(ImmutableDenseNDimArray, WithParam):
    def __new__(cls, arr: ImmutableDenseNDimArray, param_dict:dict = None) -> None:
        return ImmutableDenseNDimArray.__new__(cls, arr)
    def __init__(self, arr: ImmutableDenseNDimArray, param_dict:dict = None) -> None:
        WithParam.__init__(self, param_dict)
        
    @property
    def free_symbols(self) -> sympy.sets.sets.FiniteSet: 
        return ImmutableDenseNDimArray.free_symbols.fget(self)

    @property
    def free_params(self) -> sympy.sets.sets.FiniteSet: 
        return self.free_symbols
