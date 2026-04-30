#include <iostream>
#include <fstream>
#include <exception>
#include <array>
#include <vector>
#include <string>

#include <xtensor/xcsv.hpp>

#include <mhdcxx/basicfwd.hpp>

namespace mhd {
namespace io {

inline mhd::DivVecofArr read_divertor(std::istream& stream) {
    char const delimiter = ',';
    std::size_t skip_rows = 1;
    xt::xtensor<dtype, 2> csv_xt = xt::load_csv<dtype>(stream, delimiter, skip_rows);
    std::size_t nseg = csv_xt.shape()[0];
    mhd::DivVecofArr div(nseg+1);

    // Make sure the divertor is closed.
    if (! ( csv_xt(nseg-1, 1) == csv_xt(0, 0) && csv_xt(nseg-1, 3) == csv_xt(0, 2) ) )
        throw std::runtime_error("The beginning point of the divertor does not match its end point. Check your data format.");
    // Make sure that each line segment coheres with the next one.
    for (std::size_t i = 0; i < nseg-1; i++)
        if ( !( csv_xt(i, 1) == csv_xt(i+1, 0) && csv_xt(i, 3) == csv_xt(i+1, 2) ) ) 
            throw std::runtime_error("The divertor data has bugs that the " +std::to_string(i)+ "th segment end point does not exactly equal with the (" +std::to_string(i+1)+ ")th segment start point.");

    for (std::size_t i = 0; i < nseg; i++)
    {
        div[i][0] = csv_xt(i, 0);
        div[i][1] = csv_xt(i, 2);
    }
    div[nseg][0] = csv_xt(0, 0);
    div[nseg][0] = csv_xt(0, 2);
    return div;
};

inline mhd::DivVecofArr read_divertor(std::string const& filename) {
    std::ifstream stream(filename);
    if (!stream)
    {
        throw std::runtime_error{"io error: failed to open the divertor file: " + filename};
    }
    return mhd::io::read_divertor(stream);
};


} // close io namespace
} // close mhd namespace