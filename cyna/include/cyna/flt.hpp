#pragma once

#include <iostream>
#include <cassert>
#include <random>
#include <cmath>

#include <xtensor/xmath.hpp>
#include <ascent/Ascent.h>

#include <mhdcxx/basicfwd.hpp>
#include <mhdcxx/interpolate.hpp>
#include <mhdcxx/io.hpp>

namespace mhd
{
namespace flt
{


// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
inline bool _onSegment(mhd::Point2D p, mhd::Point2D q, mhd::Point2D r)
{
    if (q[0] <= std::max(p[0], r[0]) && q[0] >= std::min(p[0], r[0]) &&
            q[1] <= std::max(p[1], r[1]) && q[1] >= std::min(p[1], r[1]))
        return true;
    return false;
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
inline int _orientation(mhd::Point2D p, mhd::Point2D q, mhd::Point2D r, mhd::dtype tolerance=1e-8) {
    mhd::dtype val = (q[1] - p[1]) * (r[0] - q[0]) -
            (q[0] - p[0]) * (r[1] - q[1]);
    
    if (std::abs(val) < tolerance) return 0; // colinear
    return (val > 0) ? 1 : 2; // clock or counterclock wise
}
 
// The function that returns true if line segment 'p1q1' and 'p2q2' intersect.
bool _doIntersect(mhd::Point2D p1, mhd::Point2D q1, mhd::Point2D p2, mhd::Point2D q2) {
    // Find the four orientations needed for general and
    // special cases
    int o1 = _orientation(p1, q1, p2);
    int o2 = _orientation(p1, q1, q2);
    int o3 = _orientation(p2, q2, p1);
    int o4 = _orientation(p2, q2, q1);
 
    // General case
    if (o1 != o2 && o3 != o4)
        return true;
 
    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && _onSegment(p1, p2, q1)) return true;
 
    // p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && _onSegment(p1, q2, q1)) return true;
 
    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && _onSegment(p2, p1, q2)) return true;
 
    // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && _onSegment(p2, q1, q2)) return true;
 
    return false; // Doesn't fall in any of the above cases
}


inline mhd::Point2D _intersectPoint(mhd::Point2D p1, mhd::Point2D p2, mhd::Point2D p3, mhd::Point2D p4) {
    /* math of two line intersecting with the other
    The first line: 
        (x2 - x1) (y - y2) = (y2 - y1) (x - x2)
    The second line:
        (x4 - x3) (y - y4) = (y4 - y3) (x - x4)
    To solve the unknown intersecting point.x, we hve 
        [(y2 - y1) (x4 - x3)- (y4 - y3) (x2 - x1)] x 
    = (x2 - x1) (x4 - x3) (y4 - y2) - x4 (x2 - x1) (y4 - y3) + x2 (x4 - x3) (y2 - y1)
    Then we calculate the intersecting point.y by 
        y = y2 + (y2 - y1)/(x2 - x1) * (x - x2)
        y = y4 + (y4 - y3)/(x4 - x3) * (x - x4)
    we pick the line with a smaller slope to lessen error.
    */

    mhd::dtype x = (p2[0] - p1[0])*(p4[0] - p3[0])*(p4[1] - p2[1]) - p4[0]*(p2[0] - p1[0])*(p4[1] - p3[1]) + p2[0]*(p4[0] - p3[0])*(p2[1] - p1[1]);

    x /= (p2[1] - p1[1]) * (p4[0] - p3[0]) - (p4[1] - p3[1]) * (p2[0] - p1[0]);
    mhd::dtype k1 = (p2[1] - p1[1]) / (p2[0] - p1[0]);
    mhd::dtype k2 = (p4[1] - p3[1]) / (p4[0] - p3[0]);

    if (std::abs(k1) > std::abs(k2)) 
        return mhd::Point2D{x, p4[1] + k2 * (x - p4[0])};
    else 
        return mhd::Point2D{x, p2[1] + k1 * (x - p2[0])};
    
}

// Returns true if the point p lies inside the polygon[] with n vertices
bool _isInside(mhd::DivVecofArr const& polygon, mhd::Point2D p) {
    static constexpr mhd::dtype INF = 5000.0;
    std::size_t n = polygon.size() - 1; // The last point of the divertor polygon is the same as its beginning point.
    
    // There must be at least 3 vertices in polygon[]
    if (n < 3) throw std::invalid_argument("The polygon to decide whether or not point p is inside shall have more than three edges.");
 
    // Create a point for line segment from p to infinite
    mhd::Point2D extreme { INF, p[1] };
 
    // Count intersections of the above line with sides of polygon
    size_t count = 0, i = 0;
    do
    {
        int next = (i+1)%n;
        // Check if the line segment from 'p' to 'extreme' intersects
        // with the line segment from 'polygon[i]' to 'polygon[next]'
        if (_doIntersect(polygon[i], polygon[next], p, extreme))
        {
            // If the point 'p' is colinear with line segment 'i-next',
            // then check if it lies on segment. If it lies, return true,
            // otherwise false
            if (_orientation(polygon[i], p, polygon[next]) == 0)
            return _onSegment(polygon[i], p, polygon[next]);
 
            count++;
        }
        i = next;
    } while (i != 0);
 
    // Return true if count is odd, false otherwise
    return count&1; // Same as (count%2 == 1)
}

inline mhd::dtype _normdRdZ(mhd::dtype dR, mhd::dtype dZ) { return std::sqrt(std::pow(dR,2.0) + std::pow(dZ,2.0)); }

template <class E1, class E2>
asc::Recorder trace_xRZ(
    asc::state_t const init_RZ,
    mhd::interpolate::RegularGridInterpolator<2, E1, E2> const& BR_interp,
    mhd::interpolate::RegularGridInterpolator<2, E1, E2> const& BZ_interp,
    mhd::dtype dt = 0.01, mhd::dtype t_end = 10.0)
{
    auto system = [&BR_interp, &BZ_interp](asc::state_t const& x, asc::state_t& xd, mhd::dtype const /*t*/)
    {
        xd[0] = BR_interp(x);
        xd[1] = BZ_interp(x);
    };

    asc::RK4 integrator;
    asc::Recorder recorder;
    asc::state_t x = init_RZ;
    dtype t = 0.0;
    
    while (t < t_end)
    {
        recorder({ t, x[0], x[1] });
        integrator(system, x, t, dt);
        // if (streamline[f][i+1,0]<Rg.min() or streamline[f][i+1,0]>Rg.max() or streamline[f][i+1,1]<Zg.min() or streamline[f][i+1,1]>Zg.max()):
        //     print("Field line tracing out of computational domain.");
        //     streamline[f] = streamline[f][:i+1,:];
        //     break;
    }

    return recorder;
    // recorder.csv("lorenz", { "t", "x0", "x1", "x2" });
}

template <class E1, class E2>
asc::Recorder trace_xRZPhi(
    asc::state_t const init_RZPhi, 
    mhd::interpolate::RegularGridInterpolator<3, E1, E2> const& BR_interp,
    mhd::interpolate::RegularGridInterpolator<3, E1, E2> const& BZ_interp,
    mhd::interpolate::RegularGridInterpolator<3, E1, E2> const& BPhi_interp,
    mhd::DivVecofArr const& div_RZ,
    mhd::dtype dt=2.5e-2, mhd::dtype t_end = 20.0e1 )
{
    
    auto system = [&BR_interp, &BZ_interp, &BPhi_interp](asc::state_t const& x, asc::state_t& xd, mhd::dtype const /*t*/)
    {
        mhd::dtype const static PI = 3.141592653589793238463;
        asc::state_t x_mod = x;
        x_mod[2] = std::fmod( x_mod[2],  (2 * PI) ); // Note : fmod returns negative numbers if the first arg is negative while the second is positive.
        xd[0] = x[0] * BR_interp(x_mod) / BPhi_interp(x_mod);
        xd[1] = x[0] * BZ_interp(x_mod) / BPhi_interp(x_mod);
        xd[2] = 1.0;
        // xd[0] = BR_interp(x);
        // xd[1] = BZ_interp(x);
        // xd[2] = BPhi_interp(x) / x[0];
    };

    asc::RK4 integrator;
    asc::Recorder recorder;
    asc::state_t x = init_RZPhi;
    mhd::dtype t = 0.0;

    // from intersect import intersection 
    // if mean_free_path is !None: 
    //     import random
    //     from scipy.spatial.transform import Rotation
    if ( !_isInside(div_RZ, mhd::Point2D {x[0], x[1]} ) )
        throw std::runtime_error("The field line trace initiated from a point outside of the divertor.");
    
    while (t < t_end)
    {
        recorder({ t, x[0], x[1], x[2] });
        mhd::Point2D oldx_RZ { x[0], x[1] };
        mhd::dtype old_Phi = x[2];
        integrator(system, x, t, dt);

        mhd::Point2D x_RZ { x[0], x[1] };
        // Check whether the field line has been out of the region of interest surrounded by the divertor
        if ( !_isInside(div_RZ, x_RZ) )
        {
            for (size_t i = 0; i < div_RZ.size(); i++)
                if (_doIntersect(oldx_RZ, x_RZ, div_RZ[i], div_RZ[i+1]) ) {
                    mhd::Point2D cros_RZ = _intersectPoint(oldx_RZ, x_RZ, div_RZ[i], div_RZ[i+1]);
                    mhd::dtype perc = _normdRdZ(oldx_RZ[0]-cros_RZ[0], oldx_RZ[1]-cros_RZ[1]) /  _normdRdZ(x_RZ[0]-oldx_RZ[0], x_RZ[1]-oldx_RZ[1]);
                    // mhd::dtype perc = std::abs(oldx_RZ[0] - cros_RZ[0]) / std::abs(x_RZ[0]-oldx_RZ[0]);
                    // perc shall satisfy perc>=0 && perc<=1 
                    // asc::state_t x = oldx + (x-oldx)*perc;
                    recorder({ t, 
                            oldx_RZ[0] + (x_RZ[0] - oldx_RZ[0]) * perc, 
                            oldx_RZ[1] + (x_RZ[1] - oldx_RZ[1]) * perc, 
                            old_Phi    + (x[2] - old_Phi) * perc });
                    return recorder;
                }
        }// else if (x[0]<Rg.min() or x[0]>Rg.max() or x[1]<Zg.min() or x[1]>Zg.max()) 
        //    break; // print("Field line tracing out of computational domain.");

        // if mean_free_path is None:
        //     strmline_side[i+1,:] = r.integrate(r.t+dt); // Sometimes it raises error that out of bounds, in fact you may need to check whether your FLT init point is suitable 
        // else:
        //     if free_path > dt:
        //         strmline_side[i+1,:] = r.integrate(r.t+dt);
        //         free_path -= dt;
        //     else: // free_path <= dt
        //         strmline_side[i+1,:] = r.integrate(r.t+free_path);
        //         free_path = random.expovariate(1.0/mean_free_path);
        //         if xt::abs(strmline_side[i+1,1]) < 1e-6 && xt::abs(strmline_side[i+1,2]) < 1e-6:
        //             if xt::abs(strmline_side[i+1,0]) < 1e-6: raise ValueError('zero vector');
        //             else: perp_vec1 = xt::cross(strmline_side[i+1,:], [0, 1, 0]);
        //         else: perp_vec1 = xt::cross(strmline_side[i+1,:], [1, 0, 0]);
        //         perp_vec1 /= xt::linalg.norm(perp_vec1);
        //         perp_vec2 = xt::cross( strmline_side[i+1,:], perp_vec1 );
        //         perp_vec2 /= xt::linalg.norm(perp_vec2);
        //         diff_len, diff_angle = random.uniform(0.0, diffusion_range), random.uniform(0.0, 2*xt::pi);
        //         perp_vec = perp_vec1 * xt::cos(diff_angle) + perp_vec2 * xt::sin(diff_angle);
        //         perp_vec *= diff_len;
        //         strmline_side[i+1,:] += perp_vec;
        //         r.set_initial_value(strmline_side[i+1,:], t=r.t);

            
    }


    return recorder; 
}

} // close flt namespace
} // close mhd namespce

