#include <iostream>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor-io/xnpz.hpp>


int main(int argc, char *argv[])
{
    // load npz file containing multiple arrays
    auto npz_map = xt::load_npz(argv[1]);

    auto R = npz_map["R"].cast<double>();
    auto Z = npz_map["Z"].cast<double>();
    auto BR = npz_map["BR"].cast<double>();
    auto BZ = npz_map["BZ"].cast<double>();
    auto Bt = npz_map["Bt"].cast<double>();

    xt::dump_npz(argv[1]);
    return 0;
}