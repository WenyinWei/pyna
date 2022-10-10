import numpy as np
import itertools

class Poly2d:
    def __init__(self, poly2d_arr, ):
        self._arr = np.copy( poly2d_arr )
    
    
    def __call__(self, xi, highest_ord=None): # TODO: to speed up this function by memorization
        xR = xi[...,0]
        xZ = xi[...,1]
        ans = np.zeros_like(xR)
        for Rord in range(self._arr.shape[0]):
            for Zord in range(self._arr.shape[1]):
                if highest_ord is not None:
                    if Rord+Zord > highest_ord:
                        continue
                if self._arr[Rord, Zord] != 0.0:
                    ans += self._arr[Rord, Zord] * np.power(xR, Rord) * np.power(xZ, Zord)
        return ans
    
    def __add__(self, other):
        c1, c2 = self._arr, other._arr
        ans = np.zeros([max(c1.shape[i], c2.shape[i] ) for i in range(c1.ndim)])
        for c in [c1, c2]:
            clen1, clen2 = c.shape
            ans[:clen1, :clen2] += c
        return Poly2d(ans, )
    
    def __sub__(self, other): 
        c1, c2 = self._arr, other._arr
        ans = np.zeros([max(c1.shape[i], c2.shape[i] ) for i in range(c1.ndim)])
        
        clen1, clen2 = c1.shape
        ans[:clen1, :clen2] += c1
        clen1, clen2 = c2.shape
        ans[:clen1, :clen2] -= c2
        return Poly2d(ans, )
    
    def __mul__(self, other):
        c1, c2 = self._arr, other._arr
        ans = np.zeros([c1.shape[i]+c2.shape[i] -1 for i in range(c1.ndim)])
        for k in filter(lambda k: all(k[i]+c2.shape[i]<=ans.shape[i] for i in range(c1.ndim) ),
                        itertools.product(*[range(l) for l in ans.shape]) ):
            ans[ 
                k[0]:k[0]+c2.shape[0], 
                k[1]:k[1]+c2.shape[1]] += c1[k]*c2
        return Poly2d(ans, )
    
    def __pow__(self, pw):
        if pw==0:
            return Poly2d( np.ones([1,1]), ) 
        elif pw==1:
            return Poly2d( np.copy(self._arr), )
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
            if c_of_z_in_y[ky1, ky2] != 0.0:
                temp = c_of_z_in_y[ky1, ky2] * (y1_in_x**ky1 * y2_in_x**ky2)._arr
                x1_span, x2_span = temp.shape
                c_of_z_in_x[:x1_span, :x2_span] += temp
            
            
        return Poly2d( c_of_z_in_x, )

        
def anti_diagnoal(a, offset=0):
    a_hori_len = a.shape[1]
    return np.fliplr(a).diagonal(offset = a_hori_len-1-offset) 
def set_Poly2d_anti_diagnoal(poly2d, offset, x):
    for i in range(len(x)):
        poly2d._arr[i, offset-i] = x[i]