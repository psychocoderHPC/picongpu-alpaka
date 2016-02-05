/**
 * Copyright 2016 Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <alpaka/alpaka.hpp>


namespace PMacc
{
namespace alpaka
{
    //! type for defining a extent of memory
    using MemSize = size_t;
    //! type for defining indices , e.g.  for kernel dimensions
    using IdxSize = int;

    using HostDev = ::alpaka::dev::DevCpu;

#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__))
    //! device type of the accelerator
    using AccDev = ::alpaka::dev::DevCudaRt;

    //! stream type of the accelerator device
    using AccStream = ::alpaka::stream::StreamCudaRtAsync;

    /** get type of an N-dimensional accelerator
     *
     * @tparam T_Dim number of dimensions
     * @treturn alpaka accelerator type
     */
    template<
        typename T_Dim
    >
    using Acc = ::alpaka::acc::AccGpuCudaRt<
        T_Dim,
        IdxSize
    >;
    
#elif (defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED))

    //! device type of the accelerator
    using AccDev = ::alpaka::dev::DevCpu;

    //! stream type of the accelerator device
    using AccStream = ::alpaka::stream::StreamCpuAsync;

    /** get type of an N-dimensional accelerator
     *
     * @tparam T_Dim number of dimensions
     * @treturn alpaka accelerator type
     */
    template<
        typename T_Dim
    >
    using Acc = ::alpaka::acc::AccCpuOmp2Threads<
        T_Dim,
        IdxSize
    >;    
#elif (defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED))
    //! device type of the accelerator
    using AccDev = ::alpaka::dev::DevCpu;

    //! stream type of the accelerator device
    using AccStream = ::alpaka::stream::StreamCpuAsync;

    /** get type of an N-dimensional accelerator
     *
     * @tparam T_Dim number of dimensions
     * @treturn alpaka accelerator type
     */
    template<
        typename T_Dim
    >
    using Acc = ::alpaka::acc::AccCpuThreads<
        T_Dim,
        IdxSize
    >;
#else
    #error No accelerator was selected
#endif

    
    /** create alpaka dimension type out of an scalar
     *
     * @tparam T_dim number of dimensions
     * @treturn alpaka dimension type
     */
    template<
        uint32_t T_dim
    >
    using Dim = ::alpaka::dim::DimInt<T_dim>;

} // namespace alpaka
} // namespace PMacc
