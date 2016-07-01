/**
 * Copyright 2013-2016 Rene Widera
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

#include "pmacc_types.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "nvidia/atomic.hpp"

namespace PMacc
{

/* count particles in an area
 * is not optimized, it checks any partcile position if its realy a particle
 */
template<typename T_ElemSize>
struct kernelCountParticles
{
template<class PBox, class Filter, class Mapping, typename T_Acc>
DINLINE void operator()(const T_Acc& acc,
                                     PBox pb,
                                     uint64_cu* gCounter,
                                     const Filter& filter,
                                     Mapping mapper) const
{

    namespace mapElem = mappings::elements;

    typedef typename PBox::FrameType FRAME;
    typedef typename PBox::FramePtr FramePtr;
    const uint32_t Dim = Mapping::Dim;

    FramePtr frame;
    int localCounter = 0;
    sharedMem(counter, int);
    sharedMem(particlesInSuperCell, lcellId_t);


    typedef typename Mapping::SuperCellSize SuperCellSize;

    const DataSpace<Dim> stridedThreadIndex( DataSpace<Dim >(threadIdx) * T_ElemSize::toRT() );
    const DataSpace<Dim> superCellIdx(mapper.getSuperCellIndex(DataSpace<Dim > (blockIdx)));

    const int stridedLinearThreadIdx = DataSpaceOperations<Dim>::template map<SuperCellSize > (stridedThreadIndex);

    frame = pb.getLastFrame(superCellIdx);
    particlesInSuperCell = pb.getSuperCell(superCellIdx).getSizeLastFrame();

    if (stridedLinearThreadIdx == 0)
    {
        counter = 0;
    }
    __syncthreads();
    if (!frame.isValid())
        return; //end kernel if we have no frames

    PMacc::Array<Filter,T_ElemSize> filterArray;
    mapElem::vectorize<Dim>(
        [&]( const DataSpace<Dim>& idx )
        {
            filterArray(idx) = filter;
            filterArray(idx).setSuperCellPosition((superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
        },
        T_ElemSize::toRT()
    );

    while (frame.isValid())
    {
         mapElem::vectorize<Dim>(
            [&]( const DataSpace<Dim>& idx )
            {
                DataSpace<Dim> threadIndex( stridedThreadIndex + idx );
                const int linearThreadIdx = DataSpaceOperations<Dim>::template map<SuperCellSize > ( threadIndex );
                if (linearThreadIdx < particlesInSuperCell)
                {
                    if (filterArray(idx) (*frame, linearThreadIdx))
                        ++localCounter;
                }
            },
            T_ElemSize::toRT()
        );
        frame = pb.getPreviousFrame(frame);
        particlesInSuperCell = math::CT::volume<SuperCellSize>::type::value;
    }
    atomicAdd(&counter, localCounter,::alpaka::hierarchy::Threads());
    __syncthreads();
    if (stridedLinearThreadIdx == 0)
    {
        uint64_cu cnt = static_cast<uint64_cu>(counter);
        atomicAdd(gCounter, cnt,::alpaka::hierarchy::Blocks());
    }
}
};

struct CountParticles
{

    /** Get particle count
     *
     * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param filter filter instance which must inharid from PositionFilter
     * @return number of particles in defined area
     */
    template<uint32_t AREA, class PBuffer, class Filter, class CellDesc>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, Filter filter)
    {
        GridBuffer<uint64_cu, DIM1> counter(DataSpace<DIM1>(1));

        dim3 block(CellDesc::SuperCellSize::toRT().toDim3());

        AreaMapping<AREA, CellDesc> mapper(cellDescription);

        constexpr bool useElements = cupla::traits::IsThreadSeqAcc< cupla::AccThreadSeq >::value;
        if(useElements)
        {
            using ElemSize = typename  CellDesc::SuperCellSize;
            __cudaKernel_OPTI(kernelCountParticles<ElemSize>)
                (mapper.getGridDim(), block)
                (buffer.getDeviceParticlesBox(),
                 counter.getDeviceBuffer().getBasePointer(),
                 filter,
                 mapper);
        }
        else
        {
            using ElemSize = typename PMacc::math::CT::make_Int<CellDesc::Dim,1>::type::vector_type;
            __cudaKernel(kernelCountParticles<ElemSize>)
                (mapper.getGridDim(), block)
                (buffer.getDeviceParticlesBox(),
                 counter.getDeviceBuffer().getBasePointer(),
                 filter,
                 mapper);
        }

        counter.deviceToHost();
        return *(counter.getHostBuffer().getDataBox());
    }

    /** Get particle count
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param filter filter instance which must inharid from PositionFilter
     * @return number of particles in defined area
     */
    template< class PBuffer, class Filter, class CellDesc>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, Filter filter)
    {
        return PMacc::CountParticles::countOnDevice < CORE + BORDER + GUARD > (buffer, cellDescription, filter);
    }

    /** Get particle count
     *
     * @tparam AREA area were particles are counted (CORE, BORDER, GUARD)
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param origin local cell position (can be negative)
     * @param size local size in cells for checked volume
     * @return number of particles in defined area
     */
    template<uint32_t AREA, class PBuffer, class CellDesc, class Space>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, const Space& origin, const Space& size)
    {
        typedef bmpl::vector< typename GetPositionFilter<Space::Dim>::type > usedFilters;
        typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
        MyParticleFilter filter;
        filter.setStatus(true); /*activeate filter pipline*/
        filter.setWindowPosition(origin, size);
        return PMacc::CountParticles::countOnDevice<AREA>(buffer, cellDescription, filter);
    }

    /** Get particle count
     *
     * @param buffer source particle buffer
     * @param cellDescription instance of MappingDesction
     * @param origin local cell position (can be negative)
     * @param size local size in cells for checked volume
     * @return number of particles in defined area
     */
    template< class PBuffer, class Filter, class CellDesc, class Space>
    static uint64_cu countOnDevice(PBuffer& buffer, CellDesc cellDescription, const Space& origin, const Space& size)
    {
        return PMacc::CountParticles::countOnDevice < CORE + BORDER + GUARD > (buffer, cellDescription, origin, size);
    }

};

} //namespace PMacc
