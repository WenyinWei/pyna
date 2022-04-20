

class PolyMap2d:
    def __init__(self, cor1poly, cor2poly) -> None:
        self._cor1poly = cor1poly
        self._cor2poly = cor2poly

    def __matmul__(self, other): # composition, # z(y1(x1, x2), y2(x1, x2)) = z(y), where y=y(x)=(y1(x1,x2), y2(x1,x2))
        return PolyMap2d( 
            self._cor1poly @ ( other._cor1poly, other._cor2poly, ),
            self._cor2poly @ ( other._cor1poly, other._cor2poly, ) )
        
    def __pow__(self, pw):
        if pw==0:
            return PolyMap2d(
                Poly2d( np.array([[0,0], [1,0],]), ),
                Poly2d( np.array([[0,1], [0,0],]), ), 
            )            
        elif pw==1:
            return self
        return self**(pw-1) @ self