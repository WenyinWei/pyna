import numpy as np
import itertools

class Poly2d:
    def __init__(self, poly2d_arr, uncertain_from_ord:int=np.inf):
        self._arr = poly2d_arr
        self._uncertain_from_ord = uncertain_from_ord
        
    @property
    def uncertain_from_ord(self):
        return self._uncertain_from_ord
    
    def __add__(self, other):
        c1, c2 = self._arr, other._arr
        ans = np.zeros([max(c1.shape[i], c2.shape[i] ) for i in range(c1.ndim)])
        for c in [c1, c2]:
            clen1, clen2 = c.shape
            ans[:clen1, :clen2] += c
        return Poly2d(ans, uncertain_from_ord=min(self.uncertain_from_ord, other.uncertain_from_ord) )
    
    def __sub__(self, other): 
        c1, c2 = self._arr, other._arr
        ans = np.zeros([max(c1.shape[i], c2.shape[i] ) for i in range(c1.ndim)])
        
        clen1, clen2 = c1.shape
        ans[:clen1, :clen2] += c1
        clen1, clen2 = c2.shape
        ans[:clen1, :clen2] -= c2
        return Poly2d(ans, uncertain_from_ord=min(self.uncertain_from_ord, other.uncertain_from_ord) )
    
    def __mul__(self, other):
        c1, c2 = self._arr, other._arr
        ans = np.zeros([c1.shape[i]+c2.shape[i] -1 for i in range(c1.ndim)])
        for k in filter(lambda k: all(k[i]+c2.shape[i]<=ans.shape[i] for i in range(c1.ndim) ),
                        itertools.product(*[range(l) for l in ans.shape]) ):
            ans[ 
                k[0]:k[0]+c2.shape[0], 
                k[1]:k[1]+c2.shape[1]] += c1[k]*c2
        return Poly2d(ans, uncertain_from_ord=min(self.uncertain_from_ord, other.uncertain_from_ord) )
    
    def __pow__(self, pw):
        if pw==0:
            return Poly2d( np.ones([1,1]), uncertain_from_ord=np.inf  ) 
        elif pw==1:
            return Poly2d( np.copy(self._arr), uncertain_from_ord=self.uncertain_from_ord )
        return self**(pw-1) * self
    
    def __matmul__(self, other): # composition, # z(y1(x1, x2), y2(x1, x2)) = z(y), where y=y(x)=(y1(x1,x2), y2(x1,x2))
        c_of_z_in_y = self._arr
        y1_in_x, y2_in_x = other
        c_of_y1_in_x, c_of_y2_in_x = y1_in_x._arr, y2_in_x._arr
        
        ky1_max, ky2_max = c_of_z_in_y.shape
        ky1_max, ky2_max = ky1_max-1, ky2_max-1

        y1_kx1_max, y1_kx2_max = c_of_y1_in_x.shape
        y1_kx1_max, y1_kx2_max = y1_kx1_max-1, y1_kx2_max-1
        y2_kx1_max, y2_kx2_max = c_of_y2_in_x.shape
        y2_kx1_max, y2_kx2_max = y2_kx1_max-1, y2_kx2_max-1

        c_of_z_in_x = np.zeros([
            ky1_max*y1_kx1_max+ky2_max*y2_kx1_max+1, 
            ky1_max*y1_kx2_max+ky2_max*y2_kx2_max+1])

        for ky1, ky2 in itertools.product(*[range(l) for l in c_of_z_in_y.shape]) :
            temp = c_of_z_in_y[ky1, ky2] * (y1_in_x**ky1 * y2_in_x**ky2)._arr
            x1_span, x2_span = temp.shape
            c_of_z_in_x[:x1_span, :x2_span] += temp
            
        uncertain_ans = np.inf
        if self.uncertain_from_ord != np.inf: # if z(y1, y2) itself is not perfectly right.
            lowest_nonzero_ord = np.inf # lowest order of nonzero term of y1(x1, x2), y2(x1, x2)
            for kx1, kx2 in itertools.product(*[range(l) for l in c_of_y1_in_x.shape]) :
                if c_of_y1_in_x[kx1, kx2] != 0:
                    lowest_nonzero_ord = min(kx1+kx2, lowest_nonzero_ord)
            for kx1, kx2 in itertools.product(*[range(l) for l in c_of_y2_in_x.shape]) :
                if c_of_y2_in_x[kx1, kx2] != 0:
                    lowest_nonzero_ord = min(kx1+kx2, lowest_nonzero_ord)
            uncertain_self = lowest_nonzero_ord * self.uncertain_from_ord
            uncertain_ans = min( uncertain_ans, uncertain_self, )
        # no matter z(y1, y2) itself is perfectly right or not.
        for kx1, kx2 in itertools.product(*[range(l) for l in self._arr.shape]) :
            if kx1 != 0 or kx2 !=0: # don't be zero simultaneously
                if self._arr[kx1, kx2] != 0:
                    uncertain_ans = min( uncertain_ans, kx1*y1_in_x.uncertain_from_ord + kx2*y2_in_x.uncertain_from_ord )
            
        return Poly2d( c_of_z_in_x, uncertain_from_ord=uncertain_ans )