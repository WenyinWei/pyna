#pragma once

#include <string>
#include <stdexcept>
#include <list>
#include <tuple>
#include <vector>
#include <iostream> // debug

#include <xtensor/xexpression.hpp> // for xt::xexpression
#include <xtensor/xmath.hpp> // for basic operations like +-*/
#include <xtensor/xoperation.hpp>
#include <xtensor/xset_operation.hpp> // for xt::searchsorted
#include <xtensor/xstorage.hpp> // for xt::svector which is the type of xarray::shape() 
#include <xtensor/xview.hpp> // for xt::view
#include <xtensor/xindex_view.hpp> // for xt::filter

#include <mhdcxx/basicfwd.hpp>


namespace mhd
{
namespace interpolate
{
// Which kind of method to interpolate.
enum Method {nearest, linear};

// Interpolation on a regular grid in arbitrary dimensions
// The data must be voidined on a regular grid; the grid spacing however may be
// uneven. Linear and nearest-neighbor interpolation are supported. After
// setting up the interpolator object, the interpolation method (*linear* or
// *nearest*) may be chosen at each evaluation.
// Parameters
// ----------
// points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
//     The points voidining the regular grid in n dimensions.
// values : array_like, shape (m1, ..., mn, ...)
//     The data on the regular grid in n dimensions.
// method : str, optional
//     The method of interpolation to perform. Supported are "linear" and
//     "nearest". This parameter will become the voidault for the object's
//     ``__call__`` method. voidault is "linear".
// Notes
// -----
// Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
// avoids expensive triangulation of the input data by taking advantage of the
// regular grid structure.
// If any of `points` have a dimension of size 1, linear interpolation will
// return an array of `nan` values. Nearest-neighbor interpolation will work
// as usual in this case.
// .. versionadded:: 0.14
// Examples
// --------
// Evaluate a simple example function on the points of a 3-D grid:
// >>> from mhd.interpolate import RegularGridInterpolator
// >>> void f(x, y, z):
// ...     return 2 * x**3 + 3 * y**2 - z
// >>> x = xt::linspace(1, 4, 11)
// >>> y = xt::linspace(4, 7, 22)
// >>> z = xt::linspace(7, 9, 33)
// >>> xg, yg ,zg = xt::meshgrid(x, y, z, indexing='ij', sparse=True)
// >>> data = f(xg, yg, zg)
// ``data`` is now a 3-D array with ``data[i,j,k] = f(x[i], y[j], z[k])``.
// Next, voidine an interpolating function from this data:
// >>> my_interpolating_function = RegularGridInterpolator((x, y, z), data)
// Evaluate the interpolating function at the two points
// ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:
// >>> pts = xt::array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
// >>> my_interpolating_function(pts)
// array([ 125.80469388,  146.30069388])
// which is indeed a close approximation to
// ``[f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)]``.
// See also
// --------
// NearestNDInterpolator : Nearest neighbor interpolation on unstructured
//                         data in N dimensions
// LinearNDInterpolator : Piecewise linear interpolant on unstructured data
//                     in N dimensions
// References
// ----------
// .. [1] Python package *regulargrid* by Johannes Buchner, see
//     https://pypi.python.org/pypi/regulargrid/
// .. [2] Wikipedia, "Trilinear interpolation",
//     https://en.wikipedia.org/wiki/Trilinear_interpolation
// .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
//     and multilinear table interpolation in many dimensions." MATH.
//     COMPUT. 50.181 (1988): 189-196.
//     https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
// """
// this class is based on code originally programmed by Johannes Buchner,
// see https://github.com/JohannesBuchner/regulargrid

template <size_t N, class E1, class E2>
class RegularGridInterpolator
{
    const std::list< E1  > grid;
    const E2 values;

    

public:
    RegularGridInterpolator(
        std::list< E1 > const& grid, E2 const& values):
        // grid(std::apply([](const auto & dim_grid){return dim_grid.derived_cast();}, points)), 
        grid(grid), values(values)
        
    {
        // if (!hasattr(values, 'ndim')) // TODO: add check in the future
            // scipy: allow reasonable duck-typed values
        // this->values = values.derived_cast();
        const size_t ndim = grid.size();
        if (ndim > values.dimension())
            throw std::invalid_argument(
                "There are " + std::to_string(ndim) + " grid arrays, but values has " + std::to_string(values.dimension()) + " dimensions");

        // if (hasattr(values, 'mhd::dtype') and hasattr(values, 'astype'))
        //     if (!xt::issubmhd::dtype(values.mhd::dtype, xt::inexact))
        //         this->values = values.astype(float);

        auto grid_iter = grid.begin();
        for (size_t dim = 0; dim < ndim; ++dim) {
            auto& dim_grid = *grid_iter++;
            if ( !(xt::all(xt::diff(dim_grid) > 0.)) )
                throw std::invalid_argument(std::string("The points in dimension ") + std::to_string(dim) + " must be strictly ascending");
            if ( !(dim_grid.dimension() == 1) )
                throw std::invalid_argument(std::string("The points in dimension ") + std::to_string(dim) + " must be 1-dimensional");
            if ( !(values.shape(dim) == dim_grid.size() ) )
                throw std::invalid_argument(std::string("There are ") + std::to_string(dim_grid.size()) + " points and " + std::to_string(values.shape(dim)) + " values in dimension " + std::to_string(dim));
        }

    }

    constexpr size_t ndim() const 
    { 
        return N; 
    } 
    const xt::svector<size_t> value_shape() const 
    {
        xt::svector<size_t> shp{ };
        for (size_t dim = N; dim < values.dimension(); dim++) 
            shp.push_back(values.shape(dim));
        return shp;
    }

private:
    std::tuple<std::array<size_t, N>, std::array<mhd::dtype, N> > find_ind(std::vector<mhd::dtype> const& x) const
    {
        // find relevant edges between which xi are situated
        std::array<size_t, N> ind;
        // compute distance to lower edge in unity units
        std::array<mhd::dtype, N> norm_distance{};

        // iterate through dimensions
        auto grid_iter = grid.begin();
        for (size_t dim = 0; dim < N; dim++)
        {
            auto& dim_grid = *grid_iter++;
            int i = static_cast<int>(
                std::lower_bound(dim_grid.cbegin(), dim_grid.cend(), x[dim]) - dim_grid.cbegin()) - 1; 
            // be careful, i could be -1, size_t is not suitable for i                        
            if (i < 0) { i = 0; }
            else if (i > dim_grid.size() - 2) { i = dim_grid.size() - 2; }
            ind[dim] = i;
            norm_distance[dim] = (x[dim] - dim_grid(i)) / (dim_grid(i+1) - dim_grid(i));
        }
        return {ind, norm_distance};
    }
    

public:
    // Interpolation at a single point
    // Parameters
    // ----------
    // x : ndarray of shape (ndim)
    //     The coordinates to sample the gridded data. The shape is not forced to be (ndim), i.e., an array with larger shape is acceptable and the function would pick the first ndim elements from (ndim+).
    // method : std::string
    //     The method of interpolation to perform. Supported are "linear" and
    //     "nearest".
    template <Method method=Method::linear>
    mhd::dtype operator()(std::vector<mhd::dtype> const& x) const 
    {
        auto grid_iter = grid.begin();
        for (size_t dim = 0; dim < N; dim++)
        {
            auto& dim_grid = *grid_iter++;
            if (x[dim] < dim_grid(0) || dim_grid(dim_grid.size()-1) < x[dim])
                throw std::invalid_argument("The requested x is out of bounds in dimension " + std::to_string(dim) + '.');
        }
        
        auto [ind, norm_distance] = this->find_ind(x);
        // auto result = xt::xarray<mhd::dtype>::from_shape(this->value_shape());
        mhd::dtype result = 0.0;
        if constexpr (method == Method::linear) {
            // Iterate over all corners of the domain cube.
            for (size_t ind_shift = 0; ind_shift < pow(2, N); ind_shift++)
            {
                std::array<size_t, N> ind_dim_shift;
                std::array<size_t, N> corner_ind = ind;
                for (size_t dim = 0; dim < N; dim++) {
                    ind_dim_shift[dim] = (ind_shift >> dim) % 2;
                    corner_ind[dim] += ind_dim_shift[dim];
                }
                mhd::dtype weight = 1.0;
                for (size_t dim = 0; dim < N; dim++)
                {
                    size_t i = ind[dim];
                    size_t ci = corner_ind[dim];
                    mhd::dtype yi = norm_distance[dim];
                    weight *= (ci == i ? 1 - yi: yi);
                }
                // result += weight * xt::index_view(this->values, {corner_ind});
                if constexpr (N==1) 
                    result += weight * this->values(corner_ind[0]);
                else if constexpr (N==2)
                    result += weight * this->values(corner_ind[0], corner_ind[1]);
                else if constexpr (N==3)
                    result += weight * this->values(corner_ind[0], corner_ind[1], corner_ind[2]);
            }
        } else if constexpr (method == Method::nearest) {
            std::array<size_t, N> ind_nearest;
            for (size_t dim = 0; dim < N; dim++)
            {
                size_t i = ind[dim];
                mhd::dtype yi = norm_distance[dim];
                ind_nearest[dim] = (yi <= .5 ? i: i + 1);
            }
            // result = xt::index_view(this->values, {ind_nearest});
            result = this->values(ind_nearest);
            if constexpr (N==1) 
                result = this->values(ind_nearest[0]);
            else if constexpr (N==2)
                result = this->values(ind_nearest[0], ind_nearest[1]);
            else if constexpr (N==3)
                result = this->values(ind_nearest[0], ind_nearest[1], ind_nearest[2]);
        }
        return result;
    };
    
    template <Method method=Method::linear>
    auto operator()(std::vector<mhd::dtype> && x) const { return this->operator()<method>(x); };
};

} // close interpolate namespace 
} // close xt namespace 
