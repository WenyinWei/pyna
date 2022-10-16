from ..map import MapSympy, MapCallable
import sympy
from sympy import derive_by_array
import numpy as np

def jac_col_2_mat(jac):
    return np.array( [
        [jac[2], jac[0]], 
        [jac[3], jac[1]]])

def Jacobian_4_MapSympy(pmap: MapSympy):
    from ..withparam import ImmutableDenseNDimArrayWithParam
    return ImmutableDenseNDimArrayWithParam(
        derive_by_array( pmap.next_xi_exprs, pmap.xi_syms ), 
        pmap.param_dict
    ) 

def Jacobian_4_MapCallable(pmap):
    pass


def Jacobian_4_field_line_Poincare(pmap):
    pass

def partial_derivatives_num_until_order(k):
    if k==0:
        return 0
    else:
        a0 = 4
        d = 2
        return int( (2*a0 + (k-1)*d )*k/2 )
    
def term_added_one_more_factor(term, factor_No):
    term_with_one_more_factor = [ list(factor) for factor in term ]
    found_new_factor = False
    for ifactor, factor in enumerate(term_with_one_more_factor):    
        if factor[0] == factor_No: 
            term_with_one_more_factor[ifactor][1] += 1
            found_new_factor = True
    if not found_new_factor:
        term_with_one_more_factor.append([factor_No, 1])
        term_with_one_more_factor.sort(key=lambda a: a[0])
    return tuple( tuple( factor ) for factor in term_with_one_more_factor )

def dict_safe_add(d, term, n):
    if term in d:
        d[term] += n
    else:
        d[term] = n
        
def px0R_px0Z_terms_collected_as_dict(Rord, Zord,):
    k = Rord + Zord
    if Rord==1 and Zord ==0:
        return {
            ( (2,1), ): 1,
            ( (3,1), ): 1,
        }
    elif Rord==0 and Zord ==1:
        return {
            ( (0,1), ): 1,
            ( (1,1), ): 1,
        }
    else:
        d_ans = dict()
        if Rord >= 2 or (Rord==1 and Zord==1):
            d_Rord_minus_1 = px0R_px0Z_terms_collected_as_dict(Rord-1, Zord,)
            for term, termC in d_Rord_minus_1.items():
                
                term_with_one_more_pXRpx0R = term_added_one_more_factor(term, 2) # one more pXRpx0R (indexed 2)
                dict_safe_add(d_ans, term_with_one_more_pXRpx0R, termC)
                    
                term_with_one_more_pXZpx0R = term_added_one_more_factor(term, 3) # one more pXZpx0R (indexed 3)
                dict_safe_add(d_ans, term_with_one_more_pXZpx0R, termC)
                    
                for factor_No, factor_pw in term: # iterate every factor to apply chain rules
                    which_k_is_this_factor = None
                    for k_ in range(1, k +1):
                        if partial_derivatives_num_until_order(k_-1) <= factor_No \
                        and factor_No < partial_derivatives_num_until_order(k_):
                            which_k_is_this_factor = k_
                            break
                    factor_No_in_this_ord = factor_No - partial_derivatives_num_until_order(which_k_is_this_factor-1)
                    if (factor_No % 2) == 0: # \partial XR term
                        old_factor_Rord = int( factor_No_in_this_ord/2 )
                        new_factor_Rord = old_factor_Rord + 1 
                        factor_No_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*new_factor_Rord    
                    elif (factor_No % 2) == 1: # \partial XZ term
                        old_factor_Rord = int( (factor_No_in_this_ord-1)/2 )
                        new_factor_Rord = old_factor_Rord + 1 
                        factor_No_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*new_factor_Rord + 1 
#                     print(f"factor_No_new {factor_No_new}")
                        
                    new_term = [list(factor) for factor in term]
                    
                    for ifactor, factor in enumerate(new_term):
                        if factor[0] == factor_No: 
                            new_term[ifactor][1] -= 1 
                            if new_term[ifactor][1] == 0:
                                del new_term[ifactor]
                            break
                            
                    new_term = term_added_one_more_factor(new_term, factor_No_new)
                    dict_safe_add(d_ans, new_term, factor_pw*termC)
                        
        elif Zord >= 2:
            d_Zord_minus_1 = px0R_px0Z_terms_collected_as_dict(Rord, Zord-1,)
            for term, termC in d_Zord_minus_1.items():
                
                term_with_one_more_pXRpx0R = term_added_one_more_factor(term, 0) # one more pXRpx0Z (indexed 0)
                dict_safe_add(d_ans, term_with_one_more_pXRpx0R, termC)
                    
                term_with_one_more_pXZpx0R = term_added_one_more_factor(term, 1) # one more pXZpx0Z (indexed 1)
                dict_safe_add(d_ans, term_with_one_more_pXZpx0R, termC)
                    
                for factor_No, factor_pw in term: # iterate every factor to apply chain rules
                    which_k_is_this_factor = None
                    for k_ in range(1, k +1):
                        if partial_derivatives_num_until_order(k_-1) <= factor_No \
                        and factor_No < partial_derivatives_num_until_order(k_):
                            which_k_is_this_factor = k_
                            break
                    factor_No_in_this_ord = factor_No - partial_derivatives_num_until_order(which_k_is_this_factor-1)
                    if (factor_No % 2) == 0: # \partial XR term
                        old_factor_Rord = int( factor_No_in_this_ord/2 )
                        new_factor_Rord = old_factor_Rord  
                        factor_No_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*new_factor_Rord    
                    elif (factor_No % 2) == 1: # \partial XZ term
                        old_factor_Rord = int( (factor_No_in_this_ord-1)/2 )
                        new_factor_Rord = old_factor_Rord  
                        factor_No_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*new_factor_Rord + 1 
                        
                    new_term = [list(factor) for factor in term]
                    
                    for ifactor, factor in enumerate(new_term):
                        if factor[0] == factor_No: 
                            new_term[ifactor][1] -= 1 
                            if new_term[ifactor][1] == 0:
                                del new_term[ifactor]
                            break
                            
                    new_term = term_added_one_more_factor(new_term, factor_No_new)
                    dict_safe_add(d_ans, new_term, factor_pw*termC)
        return d_ans
    
import sparse
import numpy as np
import copy
def px0R_px0Z_terms_collected_as_a_sparse_array(Rord, Zord,): 
    """generate a super-sparse and super-high-dimensional array to store the inter-componenet relationship of high order derivatives of field line tracing. (FIXME: due to the sparse.DOK array dimension limit, this function fails for Rord+Zord>=4 cases. For details, see [ValueError: array size defined by dims is larger than the maximum possible size](https://github.com/pydata/sparse/issues/429) )

    Args:
        Rord (int): how many orders of derivative in R component
        Zord (int): how many orders of derivative in Z component

    Returns:
        sparse.COO: [
            the power of \partial X_{R} / \partial x_{0Z}, 
            the power of \partial X_{Z} / \partial x_{0Z}, 
            the power of \partial X_{R} / \partial x_{0R}, 
            the power of \partial X_{Z} / \partial x_{0R}, 
            the power of \partial^{2} X_{R} / \partial x_{0Z}^{2}, 
            the power of \partial^{2} X_{Z} / \partial x_{0Z}^{2}, ...]
    """
    k = Rord + Zord
    s_DOK = sparse.DOK( (k+1,) * partial_derivatives_num_until_order(k), dtype=np.int64 )
    if Rord==1 and Zord ==0:
        s_DOK[0,0,1,0] = 1
        s_DOK[0,0,0,1] = 1
    elif Rord==0 and Zord ==1:
        s_DOK[0,1,0,0] = 1
        s_DOK[1,0,0,0] = 1
    else:
        if Rord >= 2 or (Rord==1 and Zord==1):
            s_Rord_minus_1_COO = px0R_px0Z_terms_collected_as_a_sparse_array(Rord-1, Zord,)
            s_Rord_minus_1_COO_nonzero = s_Rord_minus_1_COO.nonzero()
            for nz_i in range( s_Rord_minus_1_COO.nnz ):
                nz_ind = [ s_Rord_minus_1_COO_nonzero[i][nz_i] for i in range(partial_derivatives_num_until_order(k-1)) ]
                ind_for_one_more_pXRpx0R = copy.deepcopy( nz_ind )
                ind_for_one_more_pXRpx0R[2] += 1 # index with one more pXRpx0R
                print( tuple(ind_for_one_more_pXRpx0R) )
                print( tuple(ind_for_one_more_pXRpx0R)+(0,)*(2*k+2) )  # FIXME: we give up sparse package, because it seems to be unable to support super-sparse and super-high-dimensional tensor
                s_DOK[tuple(ind_for_one_more_pXRpx0R)+(0,)*(2*k+2)] += s_Rord_minus_1_COO[tuple(nz_ind)]
                
                ind_for_one_more_pXZpx0R = copy.deepcopy( nz_ind )
                ind_for_one_more_pXZpx0R[3] += 1 # index with one more pXZpx0R
                s_DOK[tuple(ind_for_one_more_pXZpx0R)+(0,)*(2*k+2)] += s_Rord_minus_1_COO[tuple(nz_ind)]

                for ifactor, factor_pw in enumerate(nz_ind):
                    if factor_pw > 0:
                        which_k_is_this_factor = None
                        for k_ in range(1, k +1):
                            if partial_derivatives_num_until_order(k_-1) <= ifactor \
                            and ifactor < partial_derivatives_num_until_order(k_):
                                which_k_is_this_factor = k_
                                break
                        ifactor_in_this_ord = ifactor - partial_derivatives_num_until_order(which_k_is_this_factor-1)
                        if (ifactor % 2) == 0: # \partial XR term
                            chainrule_original_factor_Rord = int( ifactor_in_this_ord/2 )
                            chainrule_new_factor_Rord = chainrule_original_factor_Rord + 1 
                            ifactor_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*chainrule_new_factor_Rord    
                        elif (ifactor % 2) == 1: # \partial XZ term
                            chainrule_original_factor_Rord = int( (ifactor_in_this_ord-1)/2 )
                            chainrule_new_factor_Rord = chainrule_original_factor_Rord + 1 
                            ifactor_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*chainrule_new_factor_Rord + 1 
                            
                        chainrule_new_term_ind = list(nz_ind) + [0,]*(2*k+2)
                        chainrule_new_term_ind[ifactor] -= 1
                        chainrule_new_term_ind[ifactor_new] += 1
                        s_DOK[tuple(chainrule_new_term_ind)] += factor_pw * s_Rord_minus_1_COO[tuple(nz_ind)]
                        
        elif Zord>=2:
            s_Zord_minus_1_COO = px0R_px0Z_terms_collected_as_a_sparse_array(Rord, Zord-1,)
            s_Zord_minus_1_COO_nonzero = s_Zord_minus_1_COO.nonzero()
            for nz_i in range( s_Zord_minus_1_COO.nnz ):
                nz_ind = [ s_Zord_minus_1_COO_nonzero[i][nz_i] for i in range(partial_derivatives_num_until_order(k-1)) ]
                ind_for_one_more_pXRpx0Z = copy.deepcopy( nz_ind )
                ind_for_one_more_pXRpx0Z[0] += 1 # index with one more pXRpx0Z
                s_DOK[tuple(ind_for_one_more_pXRpx0Z)+(0,)*(2*k+2)] += s_Zord_minus_1_COO[tuple(nz_ind)]
                
                ind_for_one_more_pXZpx0Z = copy.deepcopy( nz_ind )
                ind_for_one_more_pXZpx0Z[1] += 1 # index with one more pXZpx0Z
                s_DOK[tuple(ind_for_one_more_pXZpx0Z)+(0,)*(2*k+2)] += s_Zord_minus_1_COO[tuple(nz_ind)]
                
                for ifactor, factor_pw in enumerate(nz_ind):
                    if factor_pw > 0:
                        which_k_is_this_factor = None
                        for k_ in range(1, k +1):
                            if partial_derivatives_num_until_order(k_-1) <= ifactor \
                            and ifactor < partial_derivatives_num_until_order(k_):
                                which_k_is_this_factor = k_
                                break
                        ifactor_in_this_ord = ifactor - partial_derivatives_num_until_order(which_k_is_this_factor-1)
                        if (ifactor % 2) == 0: # \partial XR term
                            chainrule_original_factor_Rord = int( ifactor_in_this_ord/2 )
                            chainrule_new_factor_Rord = chainrule_original_factor_Rord 
                            ifactor_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*chainrule_new_factor_Rord    
                        elif (ifactor % 2) == 1: # \partial XZ term
                            chainrule_original_factor_Rord = int( (ifactor_in_this_ord-1)/2 )
                            chainrule_new_factor_Rord = chainrule_original_factor_Rord  
                            ifactor_new = partial_derivatives_num_until_order(which_k_is_this_factor) + 2*chainrule_new_factor_Rord + 1 
                            
                        chainrule_new_term_ind = list(nz_ind) + [0,]*(2*k+2)
                        chainrule_new_term_ind[ifactor] -= 1
                        chainrule_new_term_ind[ifactor_new] += 1
                        s_DOK[tuple(chainrule_new_term_ind)] += factor_pw * s_Zord_minus_1_COO[tuple(nz_ind)]
                        
        
    return s_DOK.to_coo()