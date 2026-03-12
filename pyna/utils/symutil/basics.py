def divide_Array_Eq(eq):
    from sympy import Eq
    assert(eq.lhs.shape == eq.rhs.shape)
    eq_shape = eq.lhs.shape
    arr_order = len(eq_shape) 
    if arr_order > 1: # Not yet tested for high order tensor
        eq_list = []
        for i in range(eq_shape[0]):
            eq_list.append(
                divide_Array_Eq(Eq(eq.lhs[i], eq.rhs[i]))
            )
        return eq_list
    elif arr_order == 1:
        return [Eq(eq.lhs[i], eq.rhs[i]) for i in range(eq_shape[0])]
    else:# arr_order == 0
        return eq