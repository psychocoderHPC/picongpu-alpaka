/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
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

#include "eventSystem/EventSystem.hpp"
#include "eventSystem/streams/EventStream.hpp"
#include "eventSystem/tasks/StreamTask.hpp"
#include "types.h"

namespace PMacc
{

    template <class TYPE, unsigned DIM>
    class DeviceBuffer;

    template <class TYPE, unsigned DIM>
    class TaskCopyDeviceToDevice : public StreamTask
    {
    public:

        TaskCopyDeviceToDevice( DeviceBuffer<TYPE, DIM>& src, DeviceBuffer<TYPE, DIM>& dst) :
        StreamTask()
        {
            this->source = & src;
            this->destination =  & dst;
        }

        virtual ~TaskCopyDeviceToDevice()
        {
            notify(this->myId, COPYDEVICE2DEVICE, NULL);
        }

        bool executeIntern()
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*)
        {

        }

        virtual void init()
        {
            size_t current_size = source->getCurrentSize();
            destination->setCurrentSize(current_size);
            DataSpace<DIM> devCurrentSize = source->getCurrentDataSpace(current_size);
            if (source->is1D() && destination->is1D())
            {
                ::alpaka::mem::view::copy(
                    this->getEventStream()->getCudaStream(),
                    this->destination->getMemBufView1D(),
                    this->source->getMemBufView1D(),
                    static_cast<alpaka::MemSize>(
                        devCurrentSize.productOfComponents()
                    )
                );
            }
            else
            {
                ::alpaka::mem::view::copy(
                    this->getEventStream()->getCudaStream(),
                    this->destination->getMemBufView(),
                    this->source->getMemBufView(),
                    PMacc::algorithms::precisionCast::precisionCast<
                        alpaka::MemSize
                    >(devCurrentSize)
                );
            }

            this->activate();
        }

        std::string toString()
        {
            return "TaskCopyDeviceToDevice";
        }

    protected:

        DeviceBuffer<TYPE, DIM> *source;
        DeviceBuffer<TYPE, DIM> *destination;
    };

} //namespace PMacc
