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

#include "pmacc_types.hpp"
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
#define PIC_PMACC_CUDAPARAMS(...)                                                  \
        PMacc::startCuplaKernel<ThePMaccKernelName>(pmacc_taskKernel,pmacc_gridSize,pmacc_blockSize,pmacc_kernelSharedMem,__VA_ARGS__, pmacc_mapper); \
        PMACC_ACTIVATE_KERNEL                                                  \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * gridsize for kernel call is set by mapper
 *
 * @param block sizes of block on gpu
 * @param ... amount of shared memory for the kernel (optional)
 */
#define PIC_PMACC_CUDAKERNELCONFIG(block,...)                                 \
    dim3 pmacc_gridSize = pmacc_mapper.getGridDim();                                            \
    dim3 pmacc_blockSize = block;                                          \
    /*we need +0 if VA_ARGS is empty, because we must put in a value*/         \
    size_t pmacc_kernelSharedMem = __VA_ARGS__+0;                                    \
    PIC_PMACC_CUDAPARAMS

#define PIC_AREA_CUDAKERNELCONFIG(description,area) \
    AreaMapping<area,MappingDesc> pmacc_mapper(description);                               \
    CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(),"Crash before kernel call");          \
    PIC_PMACC_CUDAKERNELCONFIG

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * gridsize for kernel call is set by mapper
 * last argument of kernel call is add by mapper and is the mapper
 *
 * @param kernelname name of the CUDA kernel (can also used with templates etc. myKernnel<1>)
 * @param area area type for which the kernel is called
 */
#define __picKernelArea(...) {                               \
    PMacc::TaskKernel *pmacc_taskKernel = PMacc::Environment<>::get().Factory().createTaskKernel(#__VA_ARGS__);     \
    using ThePMaccKernelName = __VA_ARGS__; PIC_AREA_CUDAKERNELCONFIG

#define PIC_PMACC_CUDAPARAMS_OPTI(...)                                                  \
        PMacc::startCuplaKernel_OPTI<ThePMaccKernelName>(pmacc_taskKernel,pmacc_gridSize,pmacc_blockSize,pmacc_kernelSharedMem,__VA_ARGS__, pmacc_mapper); \
        PMACC_ACTIVATE_KERNEL                                                 \
    }   /*this is used if call is EventTask.waitforfinished();*/

#define PIC_PMACC_CUDAKERNELCONFIG_OPTI(block,...)                                 \
    dim3 pmacc_gridSize = pmacc_mapper.getGridDim();                                            \
    dim3 pmacc_blockSize = block;                                          \
    /*we need +0 if VA_ARGS is empty, because we must put in a value*/         \
    size_t pmacc_kernelSharedMem = __VA_ARGS__+0;                                    \
    PIC_PMACC_CUDAPARAMS_OPTI

#define PIC_AREA_CUDAKERNELCONFIG_OPTI(description,area) \
    AreaMapping<area,MappingDesc> pmacc_mapper(description);                               \
    CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(),"Crash before kernel call");          \
    PIC_PMACC_CUDAKERNELCONFIG_OPTI

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * gridsize for kernel call is set by mapper
 * last argument of kernel call is add by mapper and is the mapper
 *
 * @param kernelname name of the CUDA kernel (can also used with templates etc. myKernnel<1>)
 * @param area area type for which the kernel is called
 */
#define __picKernelArea_OPTI(...) {                               \
    PMacc::TaskKernel *pmacc_taskKernel = PMacc::Environment<>::get().Factory().createTaskKernel(#__VA_ARGS__);     \
    using ThePMaccKernelName = __VA_ARGS__; PIC_AREA_CUDAKERNELCONFIG_OPTI
