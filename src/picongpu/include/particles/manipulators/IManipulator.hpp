/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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
#include "particles/manipulators/IManipulator.def"

namespace picongpu
{

namespace particles
{
namespace manipulators
{

template<typename T_Base>
struct IManipulatorDevice : protected T_Base
{
    typedef T_Base Base;

    DINLINE IManipulatorDevice() = default;

    template<typename... T_Args>
    DINLINE IManipulatorDevice(const T_Args& ... args) : Base( args...)
    {
    }

    template<typename T_Particle1, typename T_Particle2, typename T_Acc>
    DINLINE void operator()(const T_Acc& acc,
                            T_Particle1& particleSpecies1, T_Particle2& particleSpecies2,
                            const bool isParticle1, const bool isParticle2)
    {
        return Base::operator()(acc, particleSpecies1, particleSpecies2, isParticle1, isParticle2);
    }
};


template<typename T_Base>
struct IManipulator : protected T_Base
{
    typedef T_Base Base;

    template<typename T_Acc>
    struct Get
    {
        typedef IManipulatorDevice<
            typename Base::template Get<T_Acc>::type
        > type;
    };

    template<typename T_Acc>
    typename Get<T_Acc>::type
    DINLINE get(const T_Acc& acc, const DataSpace<simDim>& localCellIdx) const
    {
        return Base::get(acc, localCellIdx);
    }

    HINLINE IManipulator(uint32_t currentStep) : Base(currentStep)
    {
    }
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
