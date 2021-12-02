

import numpy as np
from functools import lru_cache


def high_order_diff_of_fieldline_ODE_RZPhi(ndiff_R, ndiff_Z, method="Bruno"): # method can be chosen between "Bruno" or "brute_force"
    from sympy import Symbol, symbols, Function
    R, Z, Phi = symbols("R, Z, \phi", real=True)
    x0_R, x0_Z = symbols("x_{0R}, x_{0Z}", real=True)
    X_R, X_Z = [Function(latexstr, real=True)(x0_R, x0_Z, Phi) for latexstr in ["X_R", "X_Z"] ]
    X_Phi = Symbol("X_{\phi}", real=True)
    XRdot = Function("\dot{X}_{R}", real=True)(X_R, X_Z, X_Phi)

    ndiff = ndiff_R+ndiff_Z
    
    if method=="Bruno":
        from sympy import FiniteSet
        def _list_of_all_partitions(set_to_partition):
            def _partition_into_two_parts(set_to_split):
                set_size = len(set_to_split)
                all_partitions = []
                list_elements = list(set_to_split)
                first_ele = list_elements[0]
                for i in range( 1, 2**(set_size-1) ): # we skip i==0, because that is full set
                    first_part = [first_ele]
                    second_part = [ ]
                    for j, ele in enumerate(list_elements[1:]):
                        if (i >> j) % 2 == 0:
                            first_part.append(ele)
                        else:
                            second_part.append(ele)
                    if len(second_part)!=0:
                        all_partitions.append([FiniteSet(*first_part), FiniteSet(*second_part)])
                        # all_partitions.append(FiniteSet(
                        #     FiniteSet(*first_part), FiniteSet(*second_part))) # if you want a pure and beautiful recursion.
                    else:
                        all_partitions.append( [FiniteSet(*first_part)] )
                        # all_partitions.append(FiniteSet(
                        #     FiniteSet(*first_part) )) # if you want a pure and beautiful recursion.

                return all_partitions # FiniteSet(*all_partitions)
            all_partitions = [ [set_to_partition] ]
            binary_partitions = _partition_into_two_parts(set_to_partition)
            for bi_part in binary_partitions:
                if len(bi_part[-1])>1:
                    first_part = bi_part[0]
                    for secondpart_partition in _list_of_all_partitions(bi_part[-1]):
                        all_partitions.append([first_part] + secondpart_partition)
                else:
                    all_partitions.append(bi_part)
            return all_partitions

        from random import randrange
        from sympy import derive_by_array, Intersection, tensorproduct, tensorcontraction
        sum_over_big_pi = 0
        for small_pi in _list_of_all_partitions(  FiniteSet(*list(range(1, ndiff+1))) ):
            small_pi_size = len(small_pi)
            repeated_grad_result = XRdot
            for _ in range(small_pi_size):
                repeated_grad_result = derive_by_array(repeated_grad_result, [X_R, X_Z])
            
            term_of_small_pi = repeated_grad_result
            for iB, B in enumerate(small_pi):
                _ndiff_R = len(Intersection(  B, FiniteSet(*list(range(1, ndiff_R+1))) ))
                _ndiff_Z = len(Intersection(  B, FiniteSet(*list(range(ndiff_R+1, ndiff+1))) ))
                term_of_small_pi = tensorcontraction(tensorproduct(
                    term_of_small_pi, 
                    [X_R.diff(x0_R, _ndiff_R, x0_Z, _ndiff_Z), X_Z.diff(x0_R, _ndiff_R, x0_Z, _ndiff_Z)]), 
                (randrange(small_pi_size - iB), small_pi_size - iB) ) # use (small_pi_size - iB -1, small_pi_size - iB) if you want to use the last nable to dot product. In fact, no matter which nabla between [0, small_pi_size - iB -1] do we use, the result would not change. 
            sum_over_big_pi += term_of_small_pi

        result = sum_over_big_pi

    elif method=="brute_force":
        result = XRdot.diff(x0_R, ndiff_R, x0_Z, ndiff_Z)
    # The `result` has been prepared

    factored_result = result.factor()
    def _collect_factor_power_of_high_order_result(factored_result):
        import pandas as pd
        from sympy.core.numbers import Integer
        from sympy.core.function import Derivative
        from sympy.core.power import Pow

        # Preparing XRZ_inedx
        XRZ_index = ['C']
        for idiff in range(1, ndiff+1):
            for idiff_R in range(idiff+1):
                idiff_Z = idiff - idiff_R
                XRZ_index.append('XR['+str(idiff_R)+','+str(idiff_Z)+']')
                XRZ_index.append('XZ['+str(idiff_R)+','+str(idiff_Z)+']')
        # XRZ_index now has been prepared as ['C', 'XR[0,1]', 'XZ[0,1]', 'XR[1,0]', 'XZ[0,1]', 'XR[0,2]', 'XZ[0,2]'， 'XR[1,1]', 'XZ[1,1]', etc.]

        how_many_terms_in_result = len(factored_result.args)
        XRZ_df = pd.DataFrame(0, index=XRZ_index, columns=range(how_many_terms_in_result))
        # Collect the terms from `factored_result` and store the power number of each factor into a dataframe
        for i, term in enumerate(factored_result.args):
            for factor in term.args:
                if factor.func is Integer:
                    combinatorial_coeff = factor # Combinatorial number
                    XRZ_df.at['C', i] = combinatorial_coeff
                    continue
                
                if factor.func is Pow: # if the factor has power, record the power number in dataframe
                    factor_inside_power = factor.args[0]
                    factor_power = factor.args[1]
                else: # factor_power = 1 if there is no explicit power.
                    factor_inside_power = factor
                    factor_power = 1
                
                assert(factor_inside_power.func is Derivative)
                if factor_inside_power.args[0] == XRdot: # We don't care the ∂ XRdot/∂[R^{n_R}/Z^{n_Z}] factor and their info would not be recorded in dataframe because they can be calculated from other factors.
                    continue
                
                if factor_inside_power.args[0] == X_R:
                    XR_or_XZ = 'XR'
                elif factor_inside_power.args[0] == X_Z:
                    XR_or_XZ = 'XZ'
                
                x0_R_diff_n = 0
                x0_Z_diff_n = 0
                for xdiff in factor_inside_power.args[1:]:
                    if xdiff[0] == x0_R:
                        x0_R_diff_n = xdiff[1]
                    elif xdiff[0] == x0_Z:
                        x0_Z_diff_n = xdiff[1]
                
                XRZ_df.at[XR_or_XZ+'['+str(x0_R_diff_n)+','+str(x0_Z_diff_n)+']', i] = factor_power
            if XRZ_df.at['C', i] == 0:
                XRZ_df.at['C', i] = 1
        return XRZ_df
        # The dataframe has been prepared
    
    return factored_result, _collect_factor_power_of_high_order_result(factored_result)

class _FieldDifferenatiableRZ:
    def __init__(self, field: np.ndarray, R, Z, Phi) -> None:
        self._field = field # field of shape (field dimension, nR, nZ, nPhi)
        self._R = R
        self._Z = Z
        self._Phi = Phi
    
    @lru_cache
    def diff_RZ(self, nR:int, nZ:int):
        if nR == 0 and nZ == 0:
            return self._field.copy()
        elif nZ > 0:
            return np.gradient(self.diff_RZ(nR, nZ-1), self._Z, axis=-2, edge_order=2)
        elif nR > 0:
            return np.gradient(self.diff_RZ(nR-1, nZ), self._R, axis=-3, edge_order=2)
        else:
            raise ValueError("nR, nZ to differentiate in the R,Z axis shall be >= 0.")

    @lru_cache
    def diff_RZ_interpolator(self, nR:int, nZ:int):
        from scipy.interpolate import RegularGridInterpolator
        if nR > 0 and nZ > 0:
            return RegularGridInterpolator( 
                (self._R[nR:-nR], self._Z[nZ:-nZ], self._Phi), self.diff_RZ(nR, nZ)[...,nR:-nR, nZ:-nZ,:],
                method="linear", bounds_error=True )
        elif nR > 0 and nZ == 0:
            return RegularGridInterpolator( 
                (self._R[nR:-nR], self._Z, self._Phi), self.diff_RZ(nR, nZ)[...,nR:-nR, :,:],
                method="linear", bounds_error=True )
        elif nR == 0 and nZ > 0:
            return RegularGridInterpolator( 
                (self._R, self._Z[nZ:-nZ], self._Phi), self.diff_RZ(nR, nZ)[...,:, nZ:-nZ,:],
                method="linear", bounds_error=True )
        elif nR == 0 and nZ == 0:
            return RegularGridInterpolator( 
                (self._R, self._Z, self._Phi), self.diff_RZ(nR, nZ)[...,:,:,:],
                method="linear", bounds_error=True )
        else:
            raise ValueError("nR, nZ to differentiate in the R,Z axis shall be >= 0.")

from ..flow import FlowCallable
from scipy.integrate import solve_ivp
from functools import reduce
import operator
def RZ_partial_derivative_of_map_4_Flow_Phi_as_t(BR, BZ, BPhi, R, Z, Phi, t_span, y0, highest_order=1, *arg, **kwarg):
    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_field = _FieldDifferenatiableRZ(RBRdBPhi, R, Z, Phi)
    RBZdBPhi_field = _FieldDifferenatiableRZ(RBZdBPhi, R, Z, Phi)
    
    pflow = FlowCallable([
        lambda R_, Z_, Phi_: RBRdBPhi_field.diff_RZ_interpolator(0,0)([R_, Z_, Phi_])[0],
        lambda R_, Z_, Phi_: RBZdBPhi_field.diff_RZ_interpolator(0,0)([R_, Z_, Phi_])[0]
    ])
    dPhi = Phi[1] - Phi[0]
    fltsol = solve_ivp(lambda t, y: [lam(*y, t%(2*np.pi) ) for lam in pflow.diff_xi_lambdas()], t_span, y0, max_step=dPhi, dense_output=True, *arg, **kwarg) # in case of magneitc field, this is [R*BR/BPhi, R*BZ/BPhi] 
    partial_derivative_sols = [fltsol]

    # Give the lambda function according to a dataframe 
    def high_order_partial_derivative_diff_eq_lambda(termdf):
        termdf_index = termdf.index

        factor_name = termdf_index[-1] # +1 check the last one factor to know how many orders is the termdf
        factor_ndiff_R, factor_ndiff_Z = factor_name[3:-1].split(',')
        factor_ndiff_R, factor_ndiff_Z = int( factor_ndiff_R ), int( factor_ndiff_Z )
        ndiff = factor_ndiff_R + factor_ndiff_Z

        suborder_factor_params, sameorder_factor_params  = [ ], [ ] # list (term) of list (factor) of three parameters (factor_ndiff, factor_lookup_ind, pow_int)
        factor_XR_num, factor_XZ_num = [], [] # list (term) of total x0R x0Z differentiating times
        for term in termdf: # for each column
            suborder_factor_params.append([])
            sameorder_factor_params.append([])
            factor_XR_num.append(0)
            factor_XZ_num.append(0)
            for pow_ind, pow_int in enumerate( termdf[term][1:] ): # skip the first coeff term and the last 2(n+1) factors with same differential order (i.e. ndiff).
                if pow_int != 0:
                    factor_name = termdf_index[pow_ind+1] # +1 because we skip the first coefficient row
                    factor_ndiff_R, factor_ndiff_Z = factor_name[3:-1].split(',')
                    factor_ndiff_R, factor_ndiff_Z = int( factor_ndiff_R ), int( factor_ndiff_Z )
                    factor_RZ = factor_name[1]
                    if factor_RZ == 'R':
                        factor_XR_num[-1] += pow_int
                    elif factor_RZ == 'Z':
                        factor_XZ_num[-1] += pow_int
                    if factor_ndiff_R+factor_ndiff_Z != ndiff:    
                        suborder_factor_params[-1].append([None,None,None])
                        suborder_factor_params[-1][-1][0] = factor_ndiff_R + factor_ndiff_Z # factor_ndiff
                        suborder_factor_params[-1][-1][1] = 2*factor_ndiff_R if factor_RZ=='R' else 2*factor_ndiff_R+1 # factor_lookup_ind
                        suborder_factor_params[-1][-1][2] = pow_int
                    else:
                        sameorder_factor_params[-1].append([None,None,None])
                        sameorder_factor_params[-1][-1][0] = factor_ndiff_R + factor_ndiff_Z # factor_ndiff
                        sameorder_factor_params[-1][-1][1] = 2*factor_ndiff_R if factor_RZ=='R' else 2*factor_ndiff_R+1 # factor_lookup_ind
                        sameorder_factor_params[-1][-1][2] = pow_int
        return suborder_factor_params, sameorder_factor_params, factor_XR_num, factor_XZ_num

    for ndiff in range(1, highest_order+1):
        termdfs = [None]*(ndiff+1)
#         diffeq_lambdas = []
        termdfs_suborder_factor_params, termdfs_sameorder_factor_params = [None]*(ndiff+1), [None]*(ndiff+1)
        termdfs_factor_XR_num, termdfs_factor_XZ_num = [None]*(ndiff+1), [None]*(ndiff+1)
        for ndiff_R in range(ndiff+1): # Totally 2(n+1) distinct partial derivatives of order n.
            ndiff_Z = ndiff - ndiff_R
            _, termdfs[ndiff_R] = high_order_diff_of_fieldline_ODE_RZPhi(ndiff_R, ndiff_Z)
            # Totally (n+1) dataframes recording how many power for each factor, ∂XR[nR,nZ]/∂φ and ∂XZ[nR,nZ]/∂φ share a same dataframe.
            # termdfs.append(termdf) # store the term dataframes in case of termdf are removed during computation

#             diffeq_lambdas.append( high_order_partial_derivative_diff_eq_lambda(ndiff, termdfs[ndiff_R]) )
#             diffeq_lambdas.append( high_order_partial_derivative_diff_eq_lambda(ndiff, termdfs[ndiff_R]) )
            termdfs_suborder_factor_params[ndiff_R], termdfs_sameorder_factor_params[ndiff_R], termdfs_factor_XR_num[ndiff_R], termdfs_factor_XZ_num[ndiff_R] = \
                high_order_partial_derivative_diff_eq_lambda(termdfs[ndiff_R])
        def high_order_evolve_diff_eqs(t, y):
            diffeq_vals = [None]*(2*(ndiff+1))
            for ndiff_R, termdf in enumerate(termdfs):
                suborder_factor_params, sameorder_factor_params = termdfs_suborder_factor_params[ndiff_R], termdfs_sameorder_factor_params[ndiff_R]
                factor_XR_num, factor_XZ_num = termdfs_factor_XR_num[ndiff_R], termdfs_factor_XZ_num[ndiff_R]
                term_vals_R, term_vals_Z = [], []
                for iterm, term in enumerate(termdf):
                    # multiply factors like ∂^{n}R*BR/BPhi/∂R^{n}
                    term_vals_R.append(
                        float(termdf[term][0]) * 
                        reduce(operator.mul, (partial_derivative_sols[factor_ndiff].sol(t)[factor_lookup_ind]**pow_int 
                            for factor_ndiff, factor_lookup_ind, pow_int in suborder_factor_params[iterm]), 1.0 ) *
                        reduce(operator.mul, (y[factor_lookup_ind]**pow_int
                            for factor_ndiff, factor_lookup_ind, pow_int in sameorder_factor_params[iterm]), 1.0 ) *
                        RBRdBPhi_field.diff_RZ_interpolator(factor_XR_num[iterm], factor_XZ_num[iterm])([*fltsol.sol(t), t%(2*np.pi) ])[0]
                    )
                    term_vals_Z.append(
                        float(termdf[term][0]) * 
                        reduce(operator.mul, (partial_derivative_sols[factor_ndiff].sol(t)[factor_lookup_ind]**pow_int 
                            for factor_ndiff, factor_lookup_ind, pow_int in suborder_factor_params[iterm]), 1.0 ) *
                        reduce(operator.mul, (y[factor_lookup_ind]**pow_int
                            for factor_ndiff, factor_lookup_ind, pow_int in sameorder_factor_params[iterm]), 1.0 ) *
                        RBZdBPhi_field.diff_RZ_interpolator(factor_XR_num[iterm], factor_XZ_num[iterm])([*fltsol.sol(t), t%(2*np.pi) ])[0]
                    )
                diffeq_vals[2*ndiff_R  ] = sum(term_vals_R)
                diffeq_vals[2*ndiff_R+1] = sum(term_vals_Z)
            # print(diffeq_vals)
            return diffeq_vals
        # We need to solve these 2(n+1) partial derivatives together since they are correlated.
        y0 = [0.0, 1.0, 1.0, 0.0] if ndiff ==1 else [0.0]*(2*(ndiff+1))
        partial_derivative_sols.append(
            solve_ivp(high_order_evolve_diff_eqs, t_span, y0, max_step=dPhi, dense_output=True, *arg, **kwarg) )
    return partial_derivative_sols

import ray 
@ray.remote
def Poincare_trace(BR, BZ, BPhi, R, Z, Phi, x0_RZPhi, Poincare_section_Phi, times):
    Phigap_from_x0_to_section = ( Poincare_section_Phi - x0_RZPhi[-1] ) % (2*np.pi) # which shall be a float falling in the range [0, 2pi)
    if times[0] >= 0 and times[1] >= 0:
        pos_trace = RZ_partial_derivative_of_map_4_Flow_Phi_as_t(
            BR, BZ, BPhi, R, Z, Phi, 
            t_span = [x0_RZPhi[-1], x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[1] ], y0=x0_RZPhi[:-1], highest_order=0, 
            t_eval = np.linspace(x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[0], x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[1], times[1]-times[0] + 1 ) )[0].y 
        return pos_trace
    elif times[0] < 0 and times[1] >= 0:
        pos_trace = RZ_partial_derivative_of_map_4_Flow_Phi_as_t(
            BR, BZ, BPhi, R, Z, Phi, 
            t_span = [x0_RZPhi[-1], x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[1] ], y0=x0_RZPhi[:-1], highest_order=0, 
            t_eval = np.linspace(x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*0, x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[1], times[1] + 1 ) )[0].y 
        neg_trace = RZ_partial_derivative_of_map_4_Flow_Phi_as_t(
            BR, BZ, BPhi, R, Z, Phi, 
            t_span = [x0_RZPhi[-1], x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[0] ], y0=x0_RZPhi[:-1], highest_order=0, 
            t_eval = np.linspace(x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*(-1), x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[0], abs(times[0]) ) )[0].y 
        return np.concatenate( (neg_trace[:,::-1], pos_trace), axis=1)
    elif times[0] < 0 and times[1] < 0:
        neg_trace = RZ_partial_derivative_of_map_4_Flow_Phi_as_t(
            BR, BZ, BPhi, R, Z, Phi, 
            t_span = [x0_RZPhi[-1], x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[0] ], y0=x0_RZPhi[:-1], highest_order=0, 
            t_eval = np.linspace(x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[-1], x0_RZPhi[-1] + Phigap_from_x0_to_section + 2*np.pi*times[0], abs(times[0]-times[1])+1 ) )[0].y 
        return neg_trace[:,::-1]

def Poincare_plot(BR, BZ, BPhi, R, Z, Phi, init_RZPhi, Poincare_section_Phi, fig, ax):
    x_trace_future_list = []
    for i in range(len(init_RZPhi)):
        x_trace_future_list.append( Poincare_trace.remote(BR, BZ, BPhi, R, Z, Phi, init_RZPhi[i,:], Poincare_section_Phi, times=[-2,50])  ) # TODO: make the 'times' setting smarter in the future
    for i in range(len(init_RZPhi)):
        x_trace = ray.get(x_trace_future_list[i])
        ax.scatter( x_trace[0,:], x_trace[1,:], s=0.3 )