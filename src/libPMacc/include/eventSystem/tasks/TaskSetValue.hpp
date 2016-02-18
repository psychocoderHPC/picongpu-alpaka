/**
 * Copyright 2013-2016 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
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

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/EnvironmentController.hpp"
#include "memory/buffers/DeviceBuffer.hpp"
#include "memory/boxes/DataBox.hpp"
#include "eventSystem/EventSystem.hpp"
#include "eventSystem/tasks/StreamTask.hpp"

#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits.hpp>


namespace PMacc
{
namespace taskSetValueHelper
{

/** define access operation for non-pointer types
 */
template<typename T_Type, bool isPointer>
struct Value
{
    typedef const T_Type type;

    HDINLINE type& operator()(type& v) const
    {
        return v;
    }
};

/** define access operation for pointer types
 *
 * access first element of a pointer
 */
template<typename T_Type>
struct Value<T_Type, true>
{
    typedef const T_Type PtrType;
    typedef const typename boost::remove_pointer<PtrType>::type type;

    HDINLINE type& operator()(PtrType v) const
    {
        return *v;
    }
};

/** Get access to a value from a pointer or reference with the same method
 */
template<typename T_Type>
HDINLINE typename Value<T_Type, boost::is_pointer<T_Type>::value >::type&
getValue(T_Type& value)
{
    typedef Value<T_Type, boost::is_pointer<T_Type>::value > Functor;
    return Functor()(value);
}

}

template<
    uint32_t T_dim
>
struct KernelSetValue
{
    static constexpr uint32_t kernelDim = T_dim;

    template<
        typename T_Acc,
        typename T_DataBox,
        typename T_ValueType
    >
    DINLINE void operator()(
        const T_Acc& acc,
        const DataSpace<T_dim>& size,
        const T_DataBox& data,
        const T_ValueType& value
    ) const
    {
        const DataSpace<T_dim> gridKernelIdx(
            ::alpaka::idx::getIdx<
                ::alpaka::Grid,
                ::alpaka::Threads
            >(
                acc
            )
        );

        // Early out to prevent out of bounds access.
        for(uint32_t d = 0; d < dim; ++d)
        {
            if(gridKernelIdx[d] >= size[d])
            {
                return;
            }
        }
        data(gridKernelIdx) = taskSetValueHelper::getValue(value);
    }
};


template <class TYPE, unsigned DIM>
class DeviceBuffer;

/** Set all cells of a GridBuffer on the device to a given value
 *
 * T_ValueType  = data type (e.g. float, float2)
 * T_dim   = dimension of the GridBuffer
 * T_isSmallValue = true if T_ValueType can be send via kernel parameter (on cuda T_ValueType must be smaller than 256 byte)
 */
template <class T_ValueType, unsigned T_dim, bool T_isSmallValue>
class TaskSetValue;

template <class T_ValueType, unsigned T_dim>
class TaskSetValueBase : public StreamTask
{
public:
    typedef T_ValueType ValueType;
    BOOST_STATIC_CONSTEXPR uint32_t dim = T_dim;

    TaskSetValueBase(DeviceBuffer<ValueType, dim>& dst, const ValueType& value) :
    StreamTask(),
    value(value)
    {
        this->destination = &dst;
    }

    virtual ~TaskSetValueBase()
    {
        notify(this->myId, SETVALUE, NULL);

    }

    virtual void init() = 0;

    bool executeIntern()
    {
        return isFinished();
    }

    void event(id_t, EventType, IEventData*)
    {
    }

protected:

    std::string toString()
    {
        return "TaskSetValue";
    }

    DeviceBuffer<ValueType, dim> *destination;
    ValueType value;
};

/** implementation for small values (<= 256byte)
 */
template <class T_ValueType, unsigned T_dim>
class TaskSetValue<T_ValueType, T_dim, true> : public TaskSetValueBase<T_ValueType, T_dim>
{
public:
    typedef T_ValueType ValueType;
    BOOST_STATIC_CONSTEXPR uint32_t dim = T_dim;

    TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value) :
    TaskSetValueBase<ValueType, dim>(dst, value)
    {
    }

    virtual ~TaskSetValue()
    {

    }

    virtual void init()
    {
        size_t current_size = this->destination->getCurrentSize();
        const DataSpace<dim> area_size(this->destination->getCurrentDataSpace(current_size));

        DataSpace<dim> blockSize(DataSpace<dim>::create(1));
        blockSize.x()=256;

        DataSpace<dim> gridSize(area_size);

        /* line wise thread blocks*/
        gridSize.x() = ceil(double(gridSize.x()) / double(blockSize.x()));

        KernelSetValue<dim> kernel;

        ::alpaka::workdiv::WorkDivMembers<
            alpaka::Dim<dim>,
            alpaka::IdxSize
        > workDiv(
            gridSize,
            blockSize,
            DataSpace<dim>::create(1)
        );

        auto const exec(
            ::alpaka::exec::create<
                alpaka::Acc<
                    alpaka::Dim<dim>
                >
            >(
                workDiv,
                kernel,
                area_size,
                this->destination->getDataBox(),
                this->value
            )
        );

        ::alpaka::stream::enqueue(
            this->getEventStream()->getCudaStream(),
            exec
        );

        this->activate();
    }
};

/** implementation for big values (>256 byte)
 *
 * This class uses CUDA memcopy to copy an instance of T_ValueType to the GPU
 * and runs a kernel which assigns this value to all cells.
 */
template <class T_ValueType, unsigned T_dim>
class TaskSetValue<T_ValueType, T_dim, false> : public TaskSetValueBase<T_ValueType, T_dim>
{
public:
    typedef T_ValueType ValueType;
    BOOST_STATIC_CONSTEXPR uint32_t dim = T_dim;

    using  MemBufValueHost = ::alpaka::mem::buf::Buf<
        alpaka::HostDev,
        ValueType, // element type
        alpaka::Dim<DIM1>,
        alpaka::MemSize // extent type
    >;

    TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value) :
    TaskSetValueBase<ValueType, dim>(dst, value), valuePointer_host(NULL)
    {
        valuePointer_host.reset(
            new MemBufValueHost(
                ::alpaka::mem::buf::alloc<
                    ValueType,
                    alpaka::MemSize
                >(
                    Environment<>::get().DeviceManager().getHostDevice(),
                    static_cast<alpaka::MemSize>(1u)
                )
            )
        );
        *::alpaka::mem::view::getPtrNative(*valuePointer_host) = this->value;
    }

    virtual ~TaskSetValue()
    {
    }

    void init()
    {
        ::alpaka::mem::view::copy(
            this->getEventStream()->getCudaStream(),
            this->destination->getMemBufView(),
            *this->valuePointer_host,
            static_cast<alpaka::MemSize>(1u)
        );

        size_t current_size = this->destination->getCurrentSize();
        const DataSpace<dim> area_size(this->destination->getCurrentDataSpace(current_size));

        DataSpace<dim> blockSize(DataSpace<dim>::create(1));
        blockSize.x()=256;

        DataSpace<dim> gridSize(area_size);

        /* line wise thread blocks*/
        gridSize.x() = ceil(double(gridSize.x()) / double(blockSize.x()));

        KernelSetValue<dim> kernel;

        ::alpaka::workdiv::WorkDivMembers<
            alpaka::Dim<dim>,
            alpaka::IdxSize
        > workDiv(
            gridSize,
            blockSize,
            DataSpace<dim>::create(1)
        );

        auto const exec(
            ::alpaka::exec::create<
                alpaka::Acc<
                    alpaka::Dim<dim>
                >
            >(
                workDiv,
                kernel,
                area_size,
                this->destination->getDataBox(),
                ::alpaka::mem::view::getPtrNative(
                    this->destination->getMemBufView()
                )
            )
        );

        ::alpaka::stream::enqueue(
            this->getEventStream()->getCudaStream(),
            exec
        );

        this->activate();
    }

private:
    std::unique_ptr<ValueType> valuePointer_host;

};

} //namespace PMacc
