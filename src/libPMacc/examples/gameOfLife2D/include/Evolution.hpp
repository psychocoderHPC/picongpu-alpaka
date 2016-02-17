/**
 * Copyright 2013-2016 Rene Widera, Marco Garten
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#include "types.hpp"
#include "math/Vector.hpp"
#include "mappings/threads/ThreadCollective.hpp"
#include "nvidia/functors/Assign.hpp"
#include "memory/boxes/CachedBox.hpp"
#include "memory/dataTypes/Mask.hpp"

namespace gol
{
    namespace kernel
    {
        using namespace PMacc;

        struct Evolution
        {
            static constexpr uint32_t kernelDim = DIM2;

            template<
                typename T_Acc,
                class BoxReadOnly,
                class BoxWriteOnly,
                class Mapping
            >
            DINLINE void
            operator()(
                const T_Acc& acc,
                const BoxReadOnly& buffRead,
                BoxWriteOnly& buffWrite,
                const uint32_t rule,
                const Mapping& mapper
            ) const
            {
                typedef typename BoxReadOnly::ValueType Type;
                typedef SuperCellDescription<
                        typename Mapping::SuperCellSize,
                        math::CT::Int< 1, 1 >,
                        math::CT::Int< 1, 1 >
                        > BlockArea;
                PMACC_AUTO(cache, CachedBox::create < 0, Type > (acc, BlockArea()));

                const Space blockIndex(::alpaka::idx::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc));
                const Space block(mapper.getSuperCellIndex(blockIndex));
                const Space blockCell = block * Mapping::SuperCellSize::toRT();
                const Space threadIndex(::alpaka::idx::getIdx<::alpaka::Block, ::alpaka::Threads>(acc));
                PMACC_AUTO(buffRead_shifted, buffRead.shift(blockCell));

                ThreadCollective<BlockArea> collective(threadIndex);

                nvidia::functors::Assign assign;
                collective(
                          assign,
                          cache,
                          buffRead_shifted
                          );
                ::alpaka::block::sync::syncBlockThreads(acc);

                Type neighbors = 0;
                for (uint32_t i = 1; i < 9; ++i)
                {
                    Space offset(Mask::getRelativeDirections<DIM2 > (i));
                    neighbors += cache(threadIndex + offset);
                }

                Type isLife = cache(threadIndex);
                isLife = (bool)(((!isLife)*(1 << (neighbors + 9))) & rule) +
                        (bool)(((isLife)*(1 << (neighbors))) & rule);

                buffWrite(blockCell + threadIndex) = isLife;
            }
        };

        struct RandomInit
        {
            static constexpr uint32_t kernelDim = DIM2;

            template<
                typename T_Acc,
                class BoxWriteOnly,
                class Mapping
            >
            DINLINE void
            operator()(
                const T_Acc& acc,
                BoxWriteOnly& buffWrite,
                const uint32_t seed,
                const float fraction,
                const Mapping& mapper
            ) const
            {
                const Space blockIndex(::alpaka::idx::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc));
                /* get position in grid in units of SuperCells from blockID */
                const Space block(mapper.getSuperCellIndex(blockIndex));
                /* convert position in unit of cells */
                const Space blockCell = block * Mapping::SuperCellSize::toRT();
                /* convert CUDA dim3 to DataSpace<DIM3> */
                const Space threadIndex(::alpaka::idx::getIdx<::alpaka::Block, ::alpaka::Threads>(acc));
                const uint32_t cellIdx = DataSpaceOperations<DIM2>::map(
                        mapper.getGridSuperCells() * Mapping::SuperCellSize::toRT(),
                        blockCell + threadIndex);

                /* get uniform random number from seed  */
                auto gen(::alpaka::rand::generator::createDefault(acc, seed, cellIdx));
                auto rng(::alpaka::rand::distribution::createUniformReal<float>(acc));

                /* write 1(white) if uniform random number 0<rng<1 is smaller than 'fraction' */
                buffWrite(blockCell + threadIndex) = (rng(gen) <= fraction);
            }
        };
    } //namespace kernel

    template<class MappingDesc>
    struct Evolution
    {
        MappingDesc mapping;
        uint32_t rule;

        Evolution(uint32_t rule) : rule(rule)
        {

        }

        void init(const MappingDesc & desc)
        {
            mapping = desc;
        }

        template<class DBox>
        void initEvolution(const DBox & writeBox, float fraction)
        {
            AreaMapping < CORE + BORDER, MappingDesc > mapper(mapping);
            GridController<DIM2>& gc = Environment<DIM2>::get().GridController();
            uint32_t seed = gc.getGlobalSize() + gc.getGlobalRank();

            __cudaKernel(kernel::RandomInit)
                    (mapper.getGridDim(), MappingDesc::SuperCellSize::toRT())
                    (
                     writeBox,
                     seed,
                     fraction,
                     mapper);
        }

        template<uint32_t Area, class DBox>
        void run(const DBox& readBox, const DBox & writeBox)
        {
            AreaMapping < Area, MappingDesc > mapper(mapping);
            __cudaKernel(kernel::Evolution)
                    (mapper.getGridDim(), MappingDesc::SuperCellSize::toRT())
                    (readBox,
                     writeBox,
                     rule,
                     mapper);
        }
    };
}



