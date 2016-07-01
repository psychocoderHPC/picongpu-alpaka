/**
 * Copyright 2014-2016 Rene Widera
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

namespace picongpu
{
namespace particles
{
namespace manipulators
{
namespace detail
{
template<typename T_ParamClass, typename T_Functor>
struct IfRelativeGlobalPositionImpl : protected T_Functor
{
    typedef T_ParamClass ParamClass;
    typedef T_Functor Functor;

    DINLINE IfRelativeGlobalPositionImpl() = default;

    DINLINE IfRelativeGlobalPositionImpl(
        const Functor& functor,
        const DataSpace<simDim>& globalCellPosition,
        const DataSpace<simDim>& globalDomainSize
    ) : Functor(functor), m_globalCellPosition(globalCellPosition), m_globalDomainSize(globalDomainSize)
    {
    }

    template<typename T_Particle1, typename T_Particle2, typename T_Acc>
    DINLINE void operator()(const T_Acc& acc,
                            T_Particle1& particle1, T_Particle2& particle2,
                            const bool isParticle1, const bool isParticle2)
    {
        float_X relativePosition = float_X(m_globalCellPosition[ParamClass::dimension]) /
            float_X(m_globalDomainSize[ParamClass::dimension]);

        const bool inRange=(ParamClass::lowerBound <= relativePosition &&
            relativePosition < ParamClass::upperBound);
        const bool particleInRange1 = isParticle1 && inRange;
        const bool particleInRange2 = isParticle2 && inRange;

        Functor::operator()(acc,
                            particle1, particle2,
                            particleInRange1, particleInRange2);

    }

private:

    DataSpace<simDim> m_globalCellPosition;
    DataSpace<simDim> m_globalDomainSize;
};
} //namespace detail

template<typename T_ParamClass, typename T_Functor>
struct IfRelativeGlobalPositionImpl : protected T_Functor
{
    template<typename T_SpeciesType>
    struct apply
    {
        typedef IfRelativeGlobalPositionImpl<T_ParamClass, typename bmpl::apply1<T_Functor,T_SpeciesType>::type > type;
    };

    HINLINE IfRelativeGlobalPositionImpl(uint32_t currentStep) : T_Functor(currentStep)
    {
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        globalDomainSize = subGrid.getGlobalDomain().size;
        localDomainOffset = subGrid.getLocalDomain().offset;
    }

    template<typename T_Acc>
    struct Get
    {
        typedef detail::IfRelativeGlobalPositionImpl<T_ParamClass, typename T_Functor::template Get<T_Acc>::type> type;
    };

    template<typename T_Acc>
    typename Get<T_Acc>::type
    DINLINE get(const T_Acc& acc, const DataSpace<simDim>& localCellIdx) const
    {
        typedef typename Get<T_Acc>::type Functor;

        return Functor(
            T_Functor::get(acc,localCellIdx),
            localDomainOffset+localCellIdx,
            globalDomainSize
        );
    }

private:

    DataSpace<simDim> localDomainOffset;
    DataSpace<simDim> globalDomainSize;
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
