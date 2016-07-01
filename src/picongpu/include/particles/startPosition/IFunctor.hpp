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
#include "particles/startPosition/MacroParticleCfg.hpp"

namespace picongpu
{

namespace particles
{
namespace startPosition
{

template<typename T_Base>
struct IFunctorDevice : public T_Base
{
    typedef T_Base Base;

    DINLINE IFunctorDevice() = default;

    template<typename... T_Args>
    DINLINE IFunctorDevice(const T_Args& ... args) : Base( args...)
    {
    }

    DINLINE floatD_X operator()(const uint32_t currentParticleIdx)
    {
        return Base::operator()(currentParticleIdx);
    }

    DINLINE MacroParticleCfg mapRealToMacroParticle(const float_X realElPerCell)
    {
        return Base::mapRealToMacroParticle(realElPerCell);
    }
};

template<typename T_Base>
struct IFunctor : protected T_Base
{
    typedef T_Base Base;

    template<typename T_Acc>
    struct Get
    {
        typedef IFunctorDevice<
            typename Base::template Get<T_Acc>::type
        > type;
    };

    template<typename T_Acc>
    typename Get<T_Acc>::type
    DINLINE get(const T_Acc& acc, const DataSpace<simDim>& totalCellOffset) const
    {
        return Base::get(acc, totalCellOffset);
    }

    HINLINE IFunctor(uint32_t currentStep) : Base(currentStep)
    {
    }
};

} //namespace startPosition
} //namespace particles
} //namespace picongpu
