/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera, Benjamin Worpitz
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

#include "dimensions/DataSpace.hpp"
#include "eventSystem/EventSystem.hpp"
#include "ppFunctions.hpp"
#include "pmacc_types.hpp"

#include <boost/preprocessor/control/if.hpp>

/* No namespace in this file since we only declare macro defines */

namespace PMacc
{
    template< typename T_KernelType, typename ... T_Args  >
    HINLINE void startCuplaKernel_ELEM(
        PMacc::TaskKernel* taskKernel,
        dim3 gridSize,
        dim3 blockSize,
        dim3 elemSize,
        size_t externSharedMem,
        T_Args ... args
    )
    {
        CUPLA_KERNEL_ELEM(T_KernelType)(gridSize,blockSize,elemSize,externSharedMem,taskKernel->getCudaStream())(args...);
    }

    template< typename T_KernelType, typename ... T_Args  >
    HINLINE void startCuplaKernel(
        PMacc::TaskKernel* taskKernel,
        dim3 gridSize,
        dim3 blockSize,
        size_t externSharedMem,
        T_Args&& ... args
    )
    {
        CUPLA_KERNEL(T_KernelType)(gridSize,blockSize,externSharedMem,taskKernel->getCudaStream())(args...);
    }

    template< typename T_KernelType, typename ... T_Args  >
    HINLINE void startCuplaKernel_OPTI(
        PMacc::TaskKernel* taskKernel,
        dim3 gridSize,
        dim3 blockSize,
        size_t externSharedMem,
        T_Args&& ... args
    )
    {
        CUPLA_KERNEL_OPTI(T_KernelType)(gridSize,blockSize,externSharedMem,taskKernel->getCudaStream())(args...);
    }
}// namespace PMacc

/*if this flag is defined all kernel calls would be checked and synchronize
 * this flag must set by the compiler or inside of the Makefile
 */
#if (PMACC_SYNC_KERNEL  == 1)
    #define CUDA_CHECK_KERNEL_MSG(...)  CUDA_CHECK_MSG(__VA_ARGS__)
#else
    /*no synchronize and check of kernel calls*/
    #define CUDA_CHECK_KERNEL_MSG(...)  ;
#endif

/** Call activate kernel from taskKernel.
 *  If PMACC_SYNC_KERNEL is 1 cudaDeviceSynchronize() is called before
 *  and after activation.
 */
#define PMACC_ACTIVATE_KERNEL                                                           \
        CUDA_CHECK_KERNEL_MSG(cudaGetLastError( ),"Last error after kernel launch");    \
        CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(),"Crash after kernel launch");     \
        pmacc_taskKernel->activateChecks();                                                   \
        CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(),"Crash after kernel activation");

/**
 * Appends kernel arguments to generated code and activates kernel task.
 *
 * @param ... parameters to pass to kernel
 */
#define PMACC_CUDAPARAMS(...)                                                  \
        PMacc::startCuplaKernel<ThePMaccKernelName>(pmacc_taskKernel,pmacc_gridSize,pmacc_blockSize,pmacc_kernelSharedMem,__VA_ARGS__); \
        PMACC_ACTIVATE_KERNEL                                                  \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * @param grid sizes of grid on gpu
 * @param block sizes of block on gpu
 * @param ... amount of shared memory for the kernel (optional)
 */
#define PMACC_CUDAKERNELCONFIG(grid,block,...)                                 \
    dim3 pmacc_gridSize = grid;                                                      \
    dim3 pmacc_blockSize = block;                                                    \
    /*we need +0 if VA_ARGS is empty, because we must put in a value*/         \
    const size_t pmacc_kernelSharedMem = __VA_ARGS__+0;                              \
    PMACC_CUDAPARAMS

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * @param kernelname name of the CUDA kernel (can also used with templates etc. myKernel<1>)
 */
#define __cudaKernel(...) {                                                      \
    CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(),"Crash before kernel call");          \
    PMacc::TaskKernel *pmacc_taskKernel = PMacc::Environment<>::get().Factory().createTaskKernel(#__VA_ARGS__);     \
    using ThePMaccKernelName = __VA_ARGS__; PMACC_CUDAKERNELCONFIG

/**
 * Appends kernel arguments to generated code and activates kernel task.
 *
 * @param ... parameters to pass to kernel
 */
#define PMACC_CUDAPARAMS_ELEM(...)                                                  \
        PMacc::startCuplaKernel_ELEM<ThePMaccKernelName>(pmacc_taskKernel,pmacc_gridSize,pmacc_blockSize,pmacc_elemSize,pmacc_kernelSharedMem,__VA_ARGS__); \
        PMACC_ACTIVATE_KERNEL                                                  \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * @param grid sizes of grid on gpu
 * @param block sizes of block on gpu
 * @param ... amount of shared memory for the kernel (optional)
 */
#define PMACC_CUDAKERNELCONFIG_ELEM(grid,block,elem,...)                             \
    dim3 pmacc_gridSize = grid;                                                      \
    dim3 pmacc_blockSize = block;                                                    \
    dim3 pmacc_elemSize = elem;                                                      \
    /*we need +0 if VA_ARGS is empty, because we must put in a value*/         \
    const size_t pmacc_kernelSharedMem = __VA_ARGS__+0;                              \
    PMACC_CUDAPARAMS_ELEM

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * @param kernelname name of the CUDA kernel (can also used with templates etc. myKernel<1>)
 */
#define __cudaKernel_ELEM(...) {                                                      \
    CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(),"Crash before kernel call");          \
    PMacc::TaskKernel *pmacc_taskKernel = PMacc::Environment<>::get().Factory().createTaskKernel(#__VA_ARGS__);     \
    using ThePMaccKernelName = __VA_ARGS__; PMACC_CUDAKERNELCONFIG_ELEM

/**
 * Appends kernel arguments to generated code and activates kernel task.
 *
 * @param ... parameters to pass to kernel
 */
#define PMACC_CUDAPARAMS_OPTI(...)                                                  \
        PMacc::startCuplaKernel_OPTI<ThePMaccKernelName>(pmacc_taskKernel,pmacc_gridSize,pmacc_blockSize,pmacc_kernelSharedMem,__VA_ARGS__); \
        PMACC_ACTIVATE_KERNEL                                                  \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * @param grid sizes of grid on gpu
 * @param block sizes of block on gpu
 * @param ... amount of shared memory for the kernel (optional)
 */
#define PMACC_CUDAKERNELCONFIG_OPTI(grid,block,...)                                 \
    dim3 pmacc_gridSize = grid;                                                      \
    dim3 pmacc_blockSize = block;                                                    \
    /*we need +0 if VA_ARGS is empty, because we must put in a value*/         \
    const size_t pmacc_kernelSharedMem = __VA_ARGS__+0;                              \
    PMACC_CUDAPARAMS_OPTI

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * @param kernelname name of the CUDA kernel (can also used with templates etc. myKernel<1>)
 */
#define __cudaKernel_OPTI(...) {                                                      \
    CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(),"Crash before kernel call");          \
    PMacc::TaskKernel *pmacc_taskKernel = PMacc::Environment<>::get().Factory().createTaskKernel(#__VA_ARGS__);     \
    using ThePMaccKernelName = __VA_ARGS__; PMACC_CUDAKERNELCONFIG_OPTI
