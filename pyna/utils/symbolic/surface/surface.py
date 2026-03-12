from sympy import Array as _Array
from functools import cached_property

from ..geometry_map import GeometryMap
class ParametricSurface(GeometryMap):
    
    def subs(self, subs_arg):
        return ParametricSurface(self._syms, self._exprs.subs(subs_arg))

    def __or__(self, other):
        from ..geometry_map.coord_transform import CoordTransform

        assert( len(other.syms) == int(self.exprs.shape.args[0]) )
        
        if isinstance(other, CoordTransform):
            return ParametricSurface(other._exprs.subs({
                other.sym(i): self.expr(i) for i in range(len(self.syms))}), 
                self._syms)
        else:
            raise TypeError("The chain succession must be a GeometryMap.")

    @cached_property
    def r_u(self):
        return self.exprs.diff(self.sym(0)).simplify().refine()
    @cached_property
    def r_v(self):
        return self.exprs.diff(self.sym(1)).simplify().refine()

    # def __str__(self):
    #     return f"A surface = {self.exprs}, with {self.u} domain {self._u_limit}, {self.v} domain {self._v_limit}."

    # 1-form, ds^2 = Edu^2 + 2Fdudv + Gdv^2
    # dA = |\vec{r}_u x \vec{r}_v |  differential area 
    @cached_property
    def E_F_G(self): 
        from silkpy.sympy_utility import dot
        r_u, r_v = self.r_u, self.r_v
        E = dot(r_u, r_u).simplify().refine()
        F = dot(r_u, r_v).simplify().refine()
        G = dot(r_v, r_v).simplify().refine()
        return E, F, G
    @cached_property
    def metric_tensor(self):
        from einsteinpy.symbolic import MetricTensor
        E, F, G = self.E_F_G
        return MetricTensor([[E, F], [F, G]], self.syms, config='ll')

    @cached_property
    def normal_vector(self):
        from silkpy.sympy_utility import cross, norm
        from einsteinpy.symbolic.vector import GenericVector
        r_u, r_v = self.r_u, self.r_v
        r_u_x_r_v = cross(r_u, r_v)
        # cross product of r_u and r_v
        return r_u_x_r_v / norm(r_u_x_r_v)

    # 2-form, 2\delta = L du^2 + 2M dudv + N dv^2
    @cached_property
    def L_M_N(self):
        from silkpy.sympy_utility import dot
        n = self.normal_vector
        r_uu = self.exprs.diff(self.sym(0), 2)
        r_uv = self.exprs.diff(self.sym(0), self.sym(1))
        r_vv = self.exprs.diff(self.sym(1), 2)
        L = dot(r_uu, n).simplify().refine()
        M = dot(r_uv, n).simplify().refine()
        N = dot(r_vv, n).simplify().refine()
        return L, M, N
    @cached_property
    def Omega(self):
        from einsteinpy.symbolic.tensor import BaseRelativityTensor
        L, M, N = self.L_M_N
        return BaseRelativityTensor(
                    [[L, M], [M, N]], 
                    self.syms, 
                    config='ll', 
                    parent_metric=self.metric_tensor # TODO: check the metric.
                )

    @cached_property
    def christoffel_symbol(self):
        from einsteinpy.symbolic import ChristoffelSymbols
        return ChristoffelSymbols.from_metric(self.metric_tensor)
    @cached_property
    def riemann_curvature_tensor(self):
        from einsteinpy.symbolic import RiemannCurvatureTensor
        return RiemannCurvatureTensor.from_christoffels(self.christoffel_symbol)

    # \omega^m_k = \sum_{i} g^{mi} \Omega_{ik}
    @cached_property
    def weingarten_matrix(self):
        from einsteinpy.symbolic.tensor import tensor_product
        from sympy import Matrix
        return Matrix(tensor_product( 
            self.metric_tensor.change_config('uu'), 
            self.Omega, i=1,j=0).tensor().simplify().refine())
    def weingarten_transform(self, vec):
        """
        Args:
        v: planar vector in tangent plane, which would be decomposed into r_u and r_v.
        """
        from sympy.solvers.solveset import linsolve
        from sympy import Matrix, symbols
        r_u, r_v = self.r_u, self.r_v

        c_ru, c_rv = symbols('c_ru, c_rv', real=True)
        solset = linsolve(Matrix(
            ((r_u[0], r_v[0], vec[0]), 
             (r_u[1], r_v[1], vec[1]), 
             (r_u[2], r_v[2], vec[2]))), (c_ru, c_rv))
        try:    
            if len(solset) != 1:
                raise RuntimeError(f"Sympy is not smart enough to decompose the v vector with r_u, r_v as the basis.\
                It found these solutions: {solset}.\
                Users need to choose from them or deduce manually, and then set it by arguments.")
        except:
            raise RuntimeError("We failed to decompose the input vec into r_u and r_v")
        else:
            c_ru, c_rv = next(iter(solset))
        omega = self.weingarten_matrix
        W_r_u = omega[0, 0] * r_u + omega[1, 0] * r_v
        W_r_v = omega[0, 1] * r_u + omega[1, 1] * r_v
        return c_ru * W_r_u + c_rv * W_r_v

    # total curvature K = det(w^i_j)
    # average curvature H = 1/2 * Tr(w^i_j)
    @cached_property
    def K_H(self):
        w = self.weingarten_matrix
        K = w[0,0] * w[1,1] - w[0,1] * w[1,0]
        H = (w[0,0] + w[1,1]) / 2
        return K.simplify().refine(), H.simplify().refine()

    @cached_property
    def prin_curvature_and_vector(self):
        from silkpy.sympy_utility import norm
        w = self.weingarten_matrix
        r_u, r_v = self.r_u, self.r_v

        eigen = w.eigenvects()

        if eigen[0][1] == 2: # if the two eigenvalues are identical
            k1 = k2 = eigen[0][0]
            er1 = eigen[0][2][0][0] * r_u +  eigen[0][2][0][1] * r_v
            e2 = e1 = er1 / norm(er1)
        else:
            k1 = eigen[0][0]
            k2 = eigen[1][0]
            er1 = eigen[0][2][0][0] * r_u +  eigen[0][2][0][1] * r_v
            e1 = er1 / norm(er1)
            er2 = eigen[1][2][0][0] * r_u +  eigen[1][2][0][1] * r_v
            e2 = er2 / norm(er2)
        k1 = k1.simplify().refine()
        e1 = e1.simplify().refine()
        k2 = k2.simplify().refine()
        e2 = e2.simplify().refine()
        return (k1, e1), (k2, e2)
        
