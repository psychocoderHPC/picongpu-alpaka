/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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
#include "eventSystem/tasks/Factory.hpp"
#include "memory/buffers/DeviceBuffer.hpp"
#include "memory/boxes/DataBox.hpp"

#include <cassert>

namespace PMacc
{

/**
 * Internal device buffer implementation.
 */
template <class TYPE, unsigned DIM>
class DeviceBufferIntern : public DeviceBuffer<TYPE, DIM>
{
public:
    using DataBuf = ::alpaka::mem::buf::Buf<
        alpaka::AccDev,
        TYPE,
        alpaka::Dim<DIM>,
        alpaka::MemSize
    >;

    using Data1DBuf = typename DeviceBuffer<TYPE, DIM>::Data1DBuf;

    using DataView = typename PMacc::DeviceBuffer<
        TYPE,
        DIM
    >::DataView;

    using MemBufCurrentSizeDevice = typename PMacc::DeviceBuffer<
        TYPE,
        DIM
    >::MemBufCurrentSizeDevice;

    typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /*! create device buffer
     * @param size extent for each dimension (in elements)
     * @param sizeOnDevice memory with the current size of the grid is stored on device
     * @param useVectorAsBase use a vector as base of the array (is not lined pitched)
     *                      if true size on device is atomaticly set to false
     */
    DeviceBufferIntern(DataSpace<DIM> size, bool sizeOnDevice = false, bool useVectorAsBase = false) :
    DeviceBuffer<TYPE, DIM>(size, size),
    sizeOnDevice(sizeOnDevice),
    useOtherMemory(false),
    offset(DataSpace<DIM>())
    {
        //create size on device before any use of setCurrentSize
        if (useVectorAsBase)
        {
            sizeOnDevice = false;
            createSizeOnDevice(sizeOnDevice);
            createFakeData();
            this->data1D = true;
        }
        else
        {
            createSizeOnDevice(sizeOnDevice);
            createData();
            this->data1D = false;
        }

    }

    DeviceBufferIntern(DeviceBuffer<TYPE, DIM>& source, DataSpace<DIM> size, DataSpace<DIM> offset, bool sizeOnDevice = false) :
    DeviceBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize()),
    sizeOnDevice(sizeOnDevice),
    offset(), /* there are no offsets if we use alpaka plain pointer views */
    useOtherMemory(true)
    {

        auto&& dataBuffer(source.getMemBufView());

        PMacc::math::Vector<alpaka::MemSize,DIM> pitchVector;
        pitchVector.x() = ::alpaka::mem::view::getPitchBytes< DIM - 1 >(dataBuffer);

        for( uint32_t d = 1 ; d < DIM; ++d)
            pitchVector[ d ] = pitchVector[ d - 1 ] * source.getPhysicalMemorySize()[ d ];

        /* create view which is used in the tasks */
        m_dataView.reset(
            new DataView(
                &(source.getDataBox()(offset)),
                Environment<>::get().DeviceManager().getAccDevice(),
                PMacc::algorithms::precisionCast::precisionCast<alpaka::MemSize>(
                    size
                ),
                pitchVector
            )
        );

        createSizeOnDevice(sizeOnDevice);
        this->data1D = false;
    }

    virtual ~DeviceBufferIntern()
    {
        __startOperation(ITask::TASK_CUDA);
    }

    void reset(bool preserveData = true)
    {
        this->setCurrentSize(Buffer<TYPE, DIM>::getDataSpace().productOfComponents());

        __startOperation(ITask::TASK_CUDA);
        if (!preserveData)
        {
            if(!useOtherMemory)
            {
                ///@todo this call ignores the vent system
                auto&& stream(Environment<>::get().TransactionManager().getEventStream(ITask::TASK_CUDA)->getCudaStream());
                ::alpaka::mem::view::set(
                    stream,
                    this->getMemBufView(),
                    0,
                    this->getDataSpace()
                );
                ::alpaka::wait::wait(stream);
            }
            else
            {
                TYPE value;
                /* using `uint8_t` for byte-wise looping through tmp var value of `TYPE` */
                uint8_t* valuePtr = (uint8_t*)&value;
                for( size_t b = 0; b < sizeof(TYPE); ++b)
                {
                    valuePtr[b] = static_cast<uint8_t>(0);
                }
                /* set value with zero-ed `TYPE` */
                setValue(value);
            }
        }
    }

    DataBoxType getDataBox()
    {
        __startOperation(ITask::TASK_CUDA);
        return DataBoxType(PitchedBox<TYPE, DIM > (getBasePointer(), offset,
                                                   this->getPhysicalMemorySize(), this->getPitch()));
    }

    TYPE* getPointer()
    {
        __startOperation(ITask::TASK_CUDA);

        if (DIM == DIM1)
        {
            return (TYPE*) (this->getBasePointer() + this->offset[0]);
        }
        else if (DIM == DIM2)
        {
            return (TYPE*) ((char*) this->getBasePointer() + this->offset[1] * this->getPitch()) + this->offset[0];
        }
        else
        {
            const size_t offsetY = this->offset[1] * this->getPitch();
            const size_t sizePlaneXY = this->getPhysicalMemorySize()[1] * this->getPitch();
            return (TYPE*) ((char*) this->getBasePointer() + this->offset[2] * sizePlaneXY + offsetY) + this->offset[0];
        }
    }

    DataSpace<DIM> getOffset() const
    {
        return offset;
    }

    bool hasCurrentSizeOnDevice() const
    {
        return sizeOnDevice;
    }

    size_t* getCurrentSizeOnDevicePointer()
    {
        __startOperation(ITask::TASK_CUDA);
        if (!sizeOnDevice)
        {
            throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
        }
        return sizeOnDevicePtr;
    }

    size_t* getCurrentSizeHostSidePointer()
    {
        __startOperation(ITask::TASK_HOST);
        return this->current_size;
    }

    TYPE* getBasePointer()
    {
        __startOperation(ITask::TASK_CUDA);
        return ::alpaka::mem::view::getPtrNative(
                this->getMemBufView()
        );
    }

    /*! Get current size of any dimension
     * @return count of current elements per dimension
     */
    virtual size_t getCurrentSize()
    {
        if (sizeOnDevice)
        {
            __startTransaction(__getTransactionEvent());
            Environment<>::get().Factory().createTaskGetCurrentSizeFromDevice(*this);
            __endTransaction().waitForFinished();
        }

        return DeviceBuffer<TYPE, DIM>::getCurrentSize();
    }

    virtual void setCurrentSize(const size_t size)
    {
        Buffer<TYPE, DIM>::setCurrentSize(size);

        if (sizeOnDevice)
        {
            Environment<>::get().Factory().createTaskSetCurrentSizeOnDevice(
                                                                            *this, size);
        }
    }

    MemBufCurrentSizeDevice const &
    getMemBufSizeAcc() const
    {
        __startOperation(ITask::TASK_CUDA);
        if(!sizeOnDevice)
        {
            throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
        }
        return *m_memBufCurrentSizeDevice;
    }

    MemBufCurrentSizeDevice &
    getMemBufSizeAcc()
    {
        __startOperation(ITask::TASK_CUDA);
        if(!sizeOnDevice)
        {
            throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
        }
        return *m_memBufCurrentSizeDevice;
    }

    DataView const &
    getMemBufView() const
    {
        __startOperation(ITask::TASK_CUDA);
        return *m_dataView;
    }

    DataView &
    getMemBufView()
    {
        __startOperation(ITask::TASK_CUDA);
        return *m_dataView;
    }

    Data1DBuf const &
    getMemBufView1D() const
    {
        __startOperation(ITask::TASK_CUDA);
        assert(m_data1DBuf);
        return *m_data1DBuf;
    }

    Data1DBuf &
    getMemBufView1D()
    {
        __startOperation(ITask::TASK_CUDA);
        assert(m_data1DBuf);
        return *m_data1DBuf;
    }

    void copyFrom(HostBuffer<TYPE, DIM>& other)
    {

        assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyHostToDevice(other, *this);

    }

    void copyFrom(DeviceBuffer<TYPE, DIM>& other)
    {

        assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyDeviceToDevice(other, *this);

    }

    size_t getPitch() const
    {
        return ::alpaka::mem::view::getPitchBytes<DIM-1>(this->getMemBufView());
    }

    virtual void setValue(const TYPE& value)
    {
        Environment<>::get().Factory().createTaskSetValue(*this, value);
    };

private:

    /*! create native array with pitched lines
     */
    void createData()
    {
        __startOperation(ITask::TASK_CUDA);

        log<ggLog::MEMORY>("Create device %1%D data: %2% MiB") % DIM % (this->getDataSpace().productOfComponents() * sizeof(TYPE) / 1024 / 1024 );

        m_dataBuf.reset(
            new DataBuf(
                ::alpaka::mem::buf::alloc<TYPE, alpaka::MemSize>(
                    Environment<>::get().DeviceManager().getAccDevice(),
                    PMacc::algorithms::precisionCast::precisionCast<alpaka::MemSize>(
                        this->getDataSpace()
                    )
                )
            )
        );

        auto&& dataBuffer(*m_dataBuf);

        PMacc::math::Vector<alpaka::MemSize,DIM> pitchVector;
        pitchVector.x() = ::alpaka::mem::view::getPitchBytes< DIM - 1 >(dataBuffer);

        for( uint32_t d = 1 ; d < DIM; ++d)
            pitchVector[ d ] = pitchVector[ d - 1 ] * this->getDataSpace()[ d ];

        /* create view which is used in the tasks */
        m_dataView.reset(
            new DataView(
                ::alpaka::mem::view::getPtrNative(dataBuffer),
                Environment<>::get().DeviceManager().getAccDevice(),
                PMacc::algorithms::precisionCast::precisionCast<alpaka::MemSize>(
                    this->getDataSpace()
                ),
                pitchVector
            )
        );

        reset(false);
    }

    /*!create 1D, 2D, 3D Array which use only a vector as base
     */
    void createFakeData()
    {
        __startOperation(ITask::TASK_CUDA);

        log<ggLog::MEMORY>("Create device 1D data: %1% MiB") % (this->getDataSpace().productOfComponents() * sizeof (TYPE) / 1024 / 1024 );

        m_data1DBuf.reset(
            new Data1DBuf(
                ::alpaka::mem::buf::alloc<TYPE, alpaka::MemSize>(
                    Environment<>::get().DeviceManager().getAccDevice(),
                    static_cast<alpaka::MemSize>(
                        this->getDataSpace().productOfComponents()
                    )
                )
            )
        );

        auto&& dataBuffer(*m_data1DBuf);

        PMacc::math::Vector<alpaka::MemSize,DIM> pitchVector;
        pitchVector.x() = this->getDataSpace()[0] * sizeof (TYPE);

        for( uint32_t d = 1 ; d < DIM; ++d)
            pitchVector[ d ] = pitchVector[ d - 1 ] * this->getDataSpace()[ d ];

        /* create view which is used in the tasks */
        m_dataView.reset(
            new DataView(
                ::alpaka::mem::view::getPtrNative(dataBuffer),
                Environment<>::get().DeviceManager().getAccDevice(),
                PMacc::algorithms::precisionCast::precisionCast<alpaka::MemSize>(
                    this->getDataSpace()
                ),
                pitchVector
            )
        );

        reset(false);
    }

    void createSizeOnDevice(bool sizeOnDevice)
    {
        __startOperation(ITask::TASK_HOST);
        sizeOnDevicePtr = NULL;

        if (sizeOnDevice)
        {
            m_memBufCurrentSizeDevice.reset(
                new MemBufCurrentSizeDevice(
                    ::alpaka::mem::buf::alloc<
                      size_t,
                      alpaka::MemSize
                    >(
                        Environment<>::get().DeviceManager().getAccDevice(),
                        static_cast<alpaka::MemSize>(1u)
                    )
                )
            );
        }
        setCurrentSize(this->getDataSpace().productOfComponents());
    }

private:
    DataSpace<DIM> offset;

    bool sizeOnDevice;
    size_t* sizeOnDevicePtr;
    bool useOtherMemory;

    /** buffer with same dimension than `DeviceBufferIntern`
     *
     * this object can be point to null_ptr
     */
    std::unique_ptr<DataBuf> m_dataBuf;

    /** 1D buffer to create fake N dimensional data
     *
     * this object can be point to null_ptr
     */
    std::unique_ptr<Data1DBuf> m_data1DBuf;

    /** buffer which is used inside PMacc
     *
     * this object is always valid
     */
    std::unique_ptr<DataView> m_dataView;

    /** buffer with the current device size
     *
     * - this object can be point to null_ptr
     * - only valid if `sizeOnDevice` is true
     */
    std::unique_ptr<MemBufCurrentSizeDevice> m_memBufCurrentSizeDevice;

};

} //namespace PMacc
