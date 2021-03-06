/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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

#include "simulation_defines.hpp"
#include "particles/startPosition/MacroParticleCfg.hpp"
#include "nvidia/rng/RNG.hpp"
#include "nvidia/rng/methods/Xor.hpp"
#include "nvidia/rng/distributions/Uniform_float.hpp"
#include "mpi/SeedPerRank.hpp"
#include "traits/GetUniqueTypeId.hpp"

namespace picongpu
{
namespace particles
{
namespace startPosition
{

namespace nvrng = nvidia::rng;

namespace detail
{
template<typename T_ParamClass, typename T_SpeciesType, typename T_Rng>
struct RandomImpl
{
    typedef T_ParamClass ParamClass;
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

    DINLINE RandomImpl() = default;

    template<typename T_Acc>
    DINLINE RandomImpl(const T_Acc& acc, const uint32_t seed, const uint32_t cellIdx)
    {
        rng = nvrng::create(typename T_Rng::MethodType(acc,seed, cellIdx), typename T_Rng::DistributionType(acc));
    }

    /** Distributes the initial particles uniformly random within the cell.
     *
     * @param rng a reference to an initialized, UNIFORM random number generator
     * @param totalNumParsPerCell the total number of particles to init for this cell
     * @param curParticle the number of this particle: [0, totalNumParsPerCell-1]
     * @return float3_X with components between [0.0, 1.0)
     */
    DINLINE floatD_X operator()(const uint32_t)
    {
        floatD_X result;
        for (uint32_t i = 0; i < simDim; ++i)
            result[i] = rng();

        return result;
    }

    /** If the particles to initialize (numParsPerCell) end up with a
     *  related particle weighting (macroWeighting) below MIN_WEIGHTING,
     *  reduce the number of particles if possible to satisfy this condition.
     *
     * @param realParticlesPerCell  the number of real particles in this cell
     * @return macroWeighting the intended weighting per macro particle
     */
    DINLINE MacroParticleCfg mapRealToMacroParticle(const float_X realParticlesPerCell)
    {
        uint32_t numParsPerCell = ParamClass::numParticlesPerCell;
        float_X macroWeighting = float_X(0.0);
        if (numParsPerCell > 0)
            macroWeighting = realParticlesPerCell / float_X(numParsPerCell);

        while (macroWeighting < MIN_WEIGHTING &&
               numParsPerCell > 0)
        {
            --numParsPerCell;
            if (numParsPerCell > 0)
                macroWeighting = realParticlesPerCell / float_X(numParsPerCell);
            else
                macroWeighting = float_X(0.0);
        }
        MacroParticleCfg macroParCfg;
        macroParCfg.weighting = macroWeighting;
        macroParCfg.numParticlesPerCell = numParsPerCell;

        return macroParCfg;
    }

protected:
    PMACC_ALIGN(rng, T_Rng);
};

} //namespace detail

template<typename T_ParamClass, typename T_SpeciesType>
struct RandomImpl
{
    typedef T_SpeciesType SpeciesType;
    HINLINE RandomImpl(uint32_t currentStep)
    {
        typedef typename SpeciesType::FrameType FrameType;

        mpi::SeedPerRank<simDim> seedPerRank;
        GlobalSeed globalSeed;
        seed = globalSeed() ^
               PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() ^
               POSITION_SEED;
        seed = seedPerRank(seed) ^ currentStep;

        const uint32_t numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        localCells = subGrid.getLocalDomain().size;
        totalGpuOffset = subGrid.getLocalDomain( ).offset;
        totalGpuOffset.y( ) += numSlides * localCells.y( );
    }

    template<typename T_Acc>
    struct Get
    {
        typedef PMacc::nvidia::rng::RNG<nvrng::methods::Xor<T_Acc>, nvrng::distributions::Uniform_float<T_Acc> > RngType;
        typedef detail::RandomImpl<T_ParamClass, T_SpeciesType, RngType> type;
    };

    template<typename T_Acc>
    typename Get<T_Acc>::type
    DINLINE get(const T_Acc& acc, const DataSpace<simDim>& totalCellOffset) const
    {
        typedef typename Get<T_Acc>::type Functor;

        const DataSpace<simDim> localCellIdx(totalCellOffset - totalGpuOffset);
        const uint32_t cellIdx = DataSpaceOperations<simDim>::map(
                                                                  localCells,
                                                                  localCellIdx);
        return Functor(acc,seed,cellIdx);

    }

    PMACC_ALIGN(seed,uint32_t);
    PMACC_ALIGN(localCells, DataSpace<simDim>);
    PMACC_ALIGN(totalGpuOffset, DataSpace<simDim>);
};

} //namespace particlesStartPosition
} //namespace particles
} //namespace picongpu
