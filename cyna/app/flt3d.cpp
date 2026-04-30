
#include <iostream>
#include <list>
#include <array>
#include <thread>
#include <queue>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xset_operation.hpp> // for xt::searchsorted
#include <xtensor/xnpy.hpp> // for xt::load_npy
#include <xtensor-io/xnpz.hpp> // for xt::load_npz

#include <BS_thread_pool.hpp>

#include <mhdcxx/interpolate.hpp>
#include <ascent/Ascent.h> // for asc::state_t
#include <mhdcxx/flt.hpp>


        

int main(int argc, char **argv) {
    // argv[0] : where this executable file is located
    // argv[1] : indicates the npz file storing 'R', 'Z', 'Phi', 'BR', 'BZ', 'BPhi'
    // argv[2] : indicates the npy file storing the initial point array [npoints, 3], 3 for [R,Z,Phi] respectively.

    // xt::xtensor<double, 1> x = xt::linspace<double>(1, 4, 11);
    // xt::xtensor<double, 1> y = xt::linspace<double>(4, 7, 22);
    // xt::xtensor<double, 1> z = xt::linspace<double>(7, 9, 33);
    // auto [xg, yg, zg] = xt::meshgrid(x, y, z);
    
    // auto data = 2 * xt::pow(xg, 3) + 3 * xt::pow(yg, 2) - zg;

    // auto my_interpolator = mhd::interpolate::RegularGridInterpolator<3, decltype(x), decltype(data)>(
    //     std::list<xt::xtensor<double, 1>>{x, y, z}, data);

    // std::cout << std::endl<< "Compare the results and do minus operation." << std::endl;
    // for (size_t i = 0; i < data.shape(0); ++i){
    //     for (size_t j = 0; j < data.shape(1); ++j) {
    //         for (size_t k = 0; k < data.shape(2)-2; ++k) {
    //             double diff =  (data(i, j, k) + data(i, j, k+1))/2 - my_interpolator.operator()<mhd::interpolate::Method::linear>( // you can also choose nearest
    //                 asc::state_t{x(i), y(j), z(k)+1.0/33})(0);
    //             std::cout << diff << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Finished comparing.\n";

    {
        auto field_npz = xt::load_npz(argv[1]); // "103950at3500ms_EFIT_plus_vacuum.npz"
        
        xt::xtensor<double, 2> initpoint_arr = xt::load_npy<double>(argv[2]); // "103950at3500ms_EFIT_plus_vacuum.npz"
        size_t initp_num = initpoint_arr.shape()[0];
        
        // std::cout << argv[0][-3] << argv[0][-2] << argv[0][-1] << std::endl;

        // for (int i = 0; i < argc; i++)
        //     printf("argv[%d] = %s\n", i, argv[i]);
        // std::cout << argv[1] << std::cout;
        // for (int i = 0; i < 3; i++)
        // {
        //     std::cout << argv[1][strlen(argv[1])-1-3+i] ;
            
        // }
        // std::cout << std::end;

        xt::xtensor<double, 1> R = field_npz["R"].cast<double>();
        xt::xtensor<double, 1> Z = field_npz["Z"].cast<double>();
        xt::xtensor<double, 1> Phi = field_npz["Phi"].cast<double>();
        xt::xtensor<double, 3> BR = field_npz["BR"].cast<double>();
        xt::xtensor<double, 3> BZ = field_npz["BZ"].cast<double>();
        xt::xtensor<double, 3> BPhi = field_npz["BPhi"].cast<double>();
        // xt::xtensor<double, 3> BRZ= xt::stack(xt::xtuple(BR, BZ), 2);
        // auto shape = BRZ.shape();
        // std::cout << "xtensor shape of BRZ: " << shape[0] << shape[1] << shape[2] << std::endl;

        mhd::interpolate::RegularGridInterpolator<3, decltype(R), decltype(BR)> BR_interp(std::list<decltype(R)>{R, Z, Phi}, BR);
        mhd::interpolate::RegularGridInterpolator<3, decltype(R), decltype(BR)> BZ_interp(std::list<decltype(R)>{R, Z, Phi}, BZ);
        mhd::interpolate::RegularGridInterpolator<3, decltype(R), decltype(BR)> BPhi_interp(std::list<decltype(R)>{R, Z, Phi}, BPhi);

        std::string div_filename = "EAST_divertor.dat";
        mhd::DivVecofArr div_RZ = mhd::io::read_divertor(div_filename);
        
        std::cout << "So far so good 2.\n";
        

        // size_t const THREAD_NUM = 32;
        std::string which_csv_to_store_flt = std::string(argv[3]);
        // for (size_t thread_id = 0; thread_id < initp_num; thread_id++)
        // {
        //     asc::Recorder recorder = mhd::flt::trace_xRZPhi(
        //         asc::state_t {
        //             initpoint_arr[thread_id, 0], 
        //             initpoint_arr[thread_id, 1], 
        //             initpoint_arr[thread_id, 2]},
        //         BR_interp, BZ_interp, BPhi_interp, div_RZ, 0.0002, 20.0);
        //     recorder.csv(
        //         which_csv_to_store_flt+std::to_string(thread_id) , { "t", "R", "Z", "Phi"}); 
        // }

        BS::thread_pool thread_pool;
        for (size_t thread_id = 0; thread_id < initp_num; thread_id++)
        {
            thread_pool.push_task(
                [&initpoint_arr, &BR_interp, &BZ_interp, &BPhi_interp, &div_RZ, &which_csv_to_store_flt, thread_id]
                {
                    asc::Recorder recorder = mhd::flt::trace_xRZPhi(
                        asc::state_t {
                            initpoint_arr[thread_id, 0], 
                            initpoint_arr[thread_id, 1], 
                            initpoint_arr[thread_id, 2]},
                        BR_interp, BZ_interp, BPhi_interp, div_RZ, 0.0002, 20.0);
                    recorder.csv(
                        which_csv_to_store_flt+std::to_string(thread_id) , { "t", "R", "Z", "Phi"}); 
                }
            );
        }

        thread_pool.wait_for_tasks();

        // asc::Recorder recorder = mhd::flt::trace_xRZPhi(
        //     asc::state_t { 2.2, -0.1, 0.0 },
        //     BR_interp, BZ_interp, BPhi_interp, div_RZ, 0.005, 12.0
        // );
        // recorder.csv(strcat(argv[2], thread_id), { "t", "R", "Z", "Phi"});
        
        std::cout << "So far so good 3.\n";
        std::cout << "\nFinished field line tracing. \n";
    }



    // {
    //     auto nonaxisym_field = xt::load_npz("nonaxisymmetric_field.npz");

    //     auto R = nonaxisym_field["R"].cast<double>();
    //     auto Z = nonaxisym_field["Z"].cast<double>();
    //     auto BR = nonaxisym_field["BR"].cast<double>();
    //     auto BZ = nonaxisym_field["BZ"].cast<double>();
    //     auto Bt = nonaxisym_field["Bt"].cast<double>();
    // }



}