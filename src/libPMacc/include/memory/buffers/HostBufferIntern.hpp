/**
 * Copyright 2013-2016 Rene Widera, Benjamin Worpitz,
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

#include "memory/buffers/HostBuffer.hpp"
#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/EventSystem.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"

#include <cassert>

namespace PMacc
{

/**
 * Internal implementation of the HostBuffer interface.
 */
template <class TYPE, unsigned DIM>
class HostBufferIntern : public HostBuffer<TYPE, DIM>
{
public:

    using DataBuf = alpaka::mem::buf::Buf<
        alpaka::HostDev,
        TYPE,
        alpaka::Dim<DIM>,
        alpaka::MemSize
    >;

    using Data1DBuf = ::alpaka::mem::view::ViewPlainPtr<
        alpaka::HostDev,
        TYPE,
        alpaka::Dim<DIM1>,
        alpaka::MemSize
     >;

    using DataView = typename PMacc::HostBuffer<
        TYPE,
        DIM
    >::DataView;

    typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /** constructor
     *
     * @param size extent for each dimension (in elements)
     */
    HostBufferIntern(DataSpace<DIM> size) :
    HostBuffer<TYPE, DIM>(size, size),
    pointer(NULL),ownPointer(true)
    {
        /* always create 1d buffer to be sure that the PMacc definition for
         * host buffer is full filled
         *
         * *PMacc definition:* a line in a host buffer is not padded
         */
        m_data1DBuf.reset(
            new ::alpaka::mem::buf::alloc<TYPE, alpaka::MemSize>(
                Environment<>::get().DeviceManager().getAccDevice(),
                static_cast<alpaka::MemSize>(
                    this->getDataSpace().productOfComponents()
                )
            )
        );

        auto&& dataBuffer(*m_data1DBuf.get());
        /* create view which is used in the tasks */
        m_dataView.reset(
            new DataView(
                ::alpaka::mem::view::getPtrNative(dataBuffer),
                *::alpaka::mem::view::getPtrDev(dataBuffer),
                PMacc::algorithms::precisionCast::precisionCast<alpaka::MemSize>(
                    this->getDataSpace()
                ),
                this->getDataSpace()[0] * sizeof (TYPE)
            )
        );

        reset(false);
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> size, DataSpace<DIM> offset=DataSpace<DIM>()) :
    HostBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize()),
    ownPointer(false)
    {
        auto&& dataBuffer(source.getMemBufView());

        PMacc::math::Vector<alpaka::MemSize,DIM> pitchVector;
        pitchVector.x() = ::alpaka::mem::view::getPitchBytes< DIM - 1 >(dataBuffer);

        for( uint32_t d = 1 ; d < DIM; ++d)
            pitchVector[ d ] = pitchVector[ d - 1 ] * source.getPhysicalMemorySize()[ d ];

        /* create view which is used in the tasks */
        m_dataView.reset(
            new DataView(
                &(source.getDataBox()(offset)), /// @todo fix me, this is a bad way
                *::alpaka::mem::view::getPtrDev(dataBuffer),
                /* @bug alpaka can not handle views correct
                 * @todo remove this workaround
                 */
                size,
                pitchVector
            )
        );
        reset(true);
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern()
    {
        __startOperation(ITask::TASK_HOST);
    }

    /*! Get pointer of memory
     * @return pointer to memory
     */
    TYPE* getBasePointer()
    {
        __startOperation(ITask::TASK_HOST);
        return this->getPointer();
    }

    TYPE* getPointer()
    {
        __startOperation(ITask::TASK_HOST);
        return *::alpaka::mem::view::getPtrNative(this->getMemBufView());
    }

    void copyFrom(DeviceBuffer<TYPE, DIM>& other)
    {
        assert(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        Environment<>::get().Factory().createTaskCopyDeviceToHost(other, *this);
    }

    void reset(bool preserveData = true)
    {
        __startOperation(ITask::TASK_HOST);
        this->setCurrentSize(this->getDataSpace().productOfComponents());
        if (!preserveData)
        {
            if(ownPointer)
            {
                ///@todo this call ignores the vent system
                auto&& stream(Environment<>::get().TransactionManager().getEventStream(ITask::TASK_CUDA)->getCudaStream());
                alpaka::mem::view::set(
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

    void setValue(const TYPE& value)
    {
        __startOperation(ITask::TASK_HOST);
        size_t current_size = this->getCurrentSize();
        PMACC_AUTO(memBox,getDataBox());
        typedef DataBoxDim1Access<DataBoxType > D1Box;
        D1Box d1Box(memBox, this->getDataSpace());
        #pragma omp parallel for
        for (size_t i = 0; i < current_size; i++)
        {
            d1Box[i] = value;
        }
    }

    DataBoxType getDataBox()
    {
        __startOperation(ITask::TASK_HOST);
        return DataBoxType(PitchedBox<TYPE, DIM > (this->getPointer(), DataSpace<DIM > (),
                                                   this->getPhysicalMemorySize(), this->getPhysicalMemorySize()[0] * sizeof (TYPE)));
    }

    DataView const &
    getMemBufView() const
    {
        __startOperation(ITask::TASK_HOST);
        return m_dataView;
    }

    DataView &
    getMemBufView()
    {
        __startOperation(ITask::TASK_HOST);
        return m_dataView;
    }

private:

    bool ownPointer;

    ///! 1D buffer to create fake N dimensional data
    std::unique_ptr<Data1DBuf> m_data1DBuf;

    /** buffer which is used inside PMacc
     *
     * this object is always valid
     */
    std::unique_ptr<DataView> m_dataView;
};

}
