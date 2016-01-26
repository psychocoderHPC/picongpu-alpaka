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

#include "eventSystem/events/CudaEvent.hpp"
#include "types.h"

namespace PMacc
{

/**
 * Wrapper for a single cuda stream.
 * Allows recording cuda events on the stream.
 */
class EventStream
{
public:

    /**
     * Constructor.
     * Creates the cudaStream_t object.
     */
    EventStream() :
        stream(
            alpaka::AccStream(
                new Environment<DIM1>::get().DeviceManager().getAccDevice()
            )
        )
    {
    }

    /**
     * Destructor.
     * Waits for the stream to finish and destroys it.
     */
    virtual ~EventStream()
    {
        //wait for all kernels in stream to finish
        alpaka::wait::wait(*stream.get());
    }

    /**
     * Returns the cudaStream_t object associated with this EventStream.
     * @return the internal cuda stream object
     */
    const alpaka::AccStream& getCudaStream() const
    {
        return *stream.get();
    }

    alpaka::AccStream& getCudaStream()
    {
        return *stream.get();
    }

    void waitOn(const CudaEvent& ev)
    {
        if (this->stream != ev.getStream())
        {
            CUDA_CHECK(cudaStreamWaitEvent(this->getCudaStream(), *ev, 0));
        }
    }

private:
    std::shared_ptr<alpaka::AccStream> stream;
};

}
