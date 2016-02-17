/**
 * Copyright 2013-2016 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "types.h"
#include "simulation_defines.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "math/Vector.hpp"
#include "eventSystem/EventSystem.hpp"

#include "debug/PIConGPUVerbose.hpp"


namespace picongpu
{
    using namespace PMacc;

    //short name for access verbose types of picongpu
    typedef PIConGPUVerbose picLog;

} //namespace picongpu

/**
 * Appends kernel arguments to generated code and activates kernel task.
 *
 * @param ... parameters to pass to kernel
 */
#define PIC_CUDAPARAMS(...)                                                    \
        auto const workDiv =                                                   \
            ::alpaka::workdiv::WorkDivMembers<                                 \
                KernelDim,                                                     \
                ::PMacc::alpaka::IdxSize                                       \
            >(                                                                 \
                gridExtent,                                                    \
                blockExtent,                                                   \
                ::PMacc::math::Vector<                                         \
                    ::PMacc::alpaka::IdxSize,                                  \
                    KernelDim::value                                           \
                >::create(1u)                                                  \
            );                                                                 \
        auto const exec(                                                       \
            ::alpaka::exec::create<                                            \
                ::PMacc::alpaka::Acc<                                          \
                    KernelDim                                                  \
                >                                                              \
            >(                                                                 \
                workDiv,                                                       \
                theOneAndOnlyKernel,                                           \
                __VA_ARGS__,                                                   \
                mapper                                                         \
            )                                                                  \
        );                                                                     \
        ::alpaka::stream::enqueue(taskKernel->getEventStream()->getCudaStream(), exec); \
        PMACC_ACTIVATE_KERNEL                                                  \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * @param block sizes of block on gpu
 */
#define PIC_CUDAKERNELCONFIG(block)                                            \
    const auto&& gridExtent(mapper.getGridDim());                              \
    const auto&& blockExtent(block);                                           \
    PIC_CUDAPARAMS

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * @param description local domain description 
 * @param area which part of the local domain should be mapped (CORE, BORDER, GUARD or a combination)
 * @param ... name of the CUDA kernel (can also used with templates etc. myKernel<1>)
 */
#define __picKernelArea(description,area, ...) {                               \
    using KernelType = __VA_ARGS__;                                            \
    const KernelType theOneAndOnlyKernel;                                      \
    using KernelDim = ::PMacc::alpaka::Dim<KernelType::kernelDim>;                   \
    AreaMapping<area,MappingDesc> mapper(description);                         \
    CUDA_CHECK_KERNEL_MSG(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getAccDevice()),"Crash before kernel call"); \
    PMacc::TaskKernel *taskKernel = ::PMacc::Environment<>::get().Factory().createTaskKernel(#__VA_ARGS__);     \
    PIC_CUDAKERNELCONFIG
