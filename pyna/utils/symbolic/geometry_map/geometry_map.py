from functools import cached_property

class GeometryMap:
    """Base class for geometry map.
    """
    def __init__(self, syms, exprs):
        from sympy import Array, oo
        if not isinstance(exprs, list) and not isinstance(exprs, Array):
            raise ValueError("The ctor arg of GeometryMap -- exprs -- should be a list of sympy expressions.")
        self._exprs = Array(exprs)

        # if not isinstance(syms, list):
        #     raise ValueError("The ctor arg of GeometryMap -- syms -- should be a list of sympy variables, or a list of (sympy.Symbol, inf, sup)")
        self._syms = []
        for sym in syms:
            if isinstance(sym, tuple):
                assert(len(sym)==3)
                self._syms.append(sym)
            else:
                self._syms.append( (sym, -oo, +oo) )

    @property
    def exprs(self):
        return self._exprs
    def expr(self, i:int):
        return self._exprs[i]
    def subs(self, *arg):
        return GeometryMap(self, self._exprs.subs(*arg), self._syms)

    @property
    def syms(self):
        return [sym[0] for sym in self._syms]
    def sym(self, i:int):
        return self._syms[i][0]

    @property
    def sym_limits(self):
        return [sym[1:] for sym in self._syms]
    def sym_limit(self, i:int):
        return self._syms[i][1:]

    @cached_property
    def jacobian(self):
        from sympy import Array
        return self.exprs.diff(
                Array(self.syms))

    def lambdified(self, *arg, **kwarg):
        from sympy import lambdify
        return lambdify(
            self.syms, 
            self.exprs.tolist(), *arg, **kwarg)