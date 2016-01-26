/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera, Benjamin Worpitz
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

//#include <cuSTL/container/view/View.hpp>
//#include <cuSTL/container/DeviceBuffer.hpp>
#include <math/vector/Int.hpp>
#include <math/vector/Size_t.hpp>
#include <memory/buffers/Buffer.hpp>
#include <types.h>

#include <stdexcept>

namespace PMacc
{
    class EventTask;

    template <class TYPE, unsigned DIM>
    class HostBuffer;

    /**
     * Interface for a DIM-dimensional Buffer of type TYPE on the device.
     *
     * @tparam TYPE datatype of the buffer
     * @tparam DIM dimension of the buffer
     */
    template <class TYPE, unsigned DIM>
    class DeviceBuffer : public Buffer<TYPE, DIM>
    {
    protected:

        template<
            typename T_DeviceType
        >
        using  MemBufCurrentSize = ::alpaka::mem::buf::Buf<
            T_DeviceType,
            size_t, // element type
            alpaka::Dim<DIM1>,
            alpaka::MemSize // extent type
        >;
        
        using  MemBufCurrentSizeDevice = MemBufCurrentSize<
            alpaka::AccDev
        >;

        using Data1DBuf = ::alpaka::mem::buf::Buf<
            alpaka::AccDev,
            TYPE,
            alpaka::Dim<DIM1>,
            alpaka::MemSize
        >;

        using DataView = ::alpaka::mem::view::ViewPlainPtr<
            alpaka::AccDev,
            TYPE,
            alpaka::Dim<DIM>,
            alpaka::MemSize
         >;

        /** constructor
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         * @param physicalMemorySize size of the physical memory (in elements)
         */
        DeviceBuffer(DataSpace<DIM> size, DataSpace<DIM> physicalMemorySize) :
        Buffer<TYPE, DIM>(size, physicalMemorySize)
        {

        }

    public:

        using Buffer<TYPE, DIM>::setCurrentSize; //!\todo :this function was hidden, I don't know why.

        /**
         * Destructor.
         */
        virtual ~DeviceBuffer()
        {
        };

/*
#define COMMA ,

        HINLINE
        container::CartBuffer<TYPE, DIM, allocator::DeviceMemAllocator<TYPE, DIM>,
                                copier::D2DCopier<DIM>,
                                assigner::DeviceMemAssigner<> >
        cartBuffer() const
        {
            container::DeviceBuffer<TYPE, DIM> result;
            auto & memBufView = getMemBufView();
            result.dataPointer = ::alpaka::mem::view::getPtrNative(memBufView);
            result._size = (math::Size_t<DIM>)this->getDataSpace();
            if(DIM == 2)
                result.pitch[0] = this->getPitch();
            if(DIM == 3)
            {
                result.pitch[0] = this->getPitch();
                result.pitch[1] = result.pitch[0] * this->getPhysicalMemorySize()[1];
            }
#ifndef __CUDA_ARCH__
            result.refCount = new int;
#endif
            *result.refCount = 2;
            return result;
        }
#undef COMMA
*/

        /**
         * Returns offset of elements in every dimension.
         *
         * @return count of elements
         */
        virtual DataSpace<DIM> getOffset() const = 0;

        /**
         * Show if current size is stored on device.
         *
         * @return return false if no size is stored on device, true otherwise
         */
        virtual bool hasCurrentSizeOnDevice() const = 0;

        /** get native alpaka buffer of the `DeviceBuffer`
         *
         * location of the alpaka buffer is DEVICE
         * @{/
         */
        virtual
        MemBufCurrentSizeDevice &
        getMemBufSizeAcc() = 0;
        /// @}

        /** Returns a view to the internal alpaka buffer.
         *
         * @return view to alpaka buffer
         *
         * @{
         */
        virtual DataView const & getMemBufView() const = 0;

        virtual DataView & getMemBufView() = 0;
        /// @}

        virtual Data1DBuf const & getMemBufView1D() const = 0;

        virtual Data1DBuf & getMemBufView1D() = 0;

        /**
         * Returns pointer to current size on device.
         *
         * @return pointer which point to device memory of current size
         */
        virtual size_t* getCurrentSizeOnDevicePointer() = 0;

        /** Returns host pointer of current size storage
         *
         * @return pointer to stored value on host side
         */
        virtual size_t* getCurrentSizeHostSidePointer()=0;

        /**
         * Sets current size of any dimension.
         *
         * If stream is 0, this function is blocking (we use a kernel to set size).
         * Keep in mind: on Fermi-architecture, kernels in different streams may run at the same time
         * (only used if size is on device).
         *
         * @param size count of elements per dimension
         */
        virtual void setCurrentSize(const size_t size) = 0;

        /** get line pitch of memory in byte
         *
         * @return size of one line in memory
         */
        virtual size_t getPitch() const = 0;

        /**
         * Copies data from the given HostBuffer to this DeviceBuffer.
         *
         * @param other the HostBuffer to copy from
         */
        virtual void copyFrom(HostBuffer<TYPE, DIM>& other) = 0;

        /**
         * Copies data from the given DeviceBuffer to this DeviceBuffer.
         *
         * @param other the DeviceBuffer to copy from
         */
        virtual void copyFrom(DeviceBuffer<TYPE, DIM>& other) = 0;

    };

} //namespace PMacc
