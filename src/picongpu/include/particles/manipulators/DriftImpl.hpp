/**
 * Copyright 2013-2016 Axel Huebl, Rene Widera
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
template<typename T_ParamClass, typename T_ValueFunctor>
struct DriftImpl : protected T_ValueFunctor
{
    typedef T_ParamClass ParamClass;
    typedef T_ValueFunctor ValueFunctor;

    DINLINE DriftImpl() = default;

    template<typename T_Particle1, typename T_Particle2, typename T_Acc>
    DINLINE void operator()(const T_Acc& acc,
                            T_Particle1& particle, T_Particle2&,
                            const bool isParticle, const bool)
    {

        if (isParticle)
        {
            const float_X macroWeighting = particle[weighting_];
            const float_X macroMass = attribute::getMass(macroWeighting,particle);

            const float_64 myGamma = ParamClass::gamma;

            const float_64 initDriftBeta =
                math::sqrt(1.0 -
                           1.0 / (myGamma *
                                  myGamma));

            const float3_X driftDirection(ParamClass().direction);
            const float3_X normDir = driftDirection /
                math::abs(driftDirection);

            const float3_X mom(normDir *
                               float_X(myGamma *
                                       initDriftBeta *
                                       float_64(macroMass) *
                                       float_64(SPEED_OF_LIGHT)
                                       ));

            ValueFunctor::operator()(particle[momentum_], mom);
        }
    }

};

} //namespace detail

template<typename T_ParamClass, typename T_ValueFunctor>
struct DriftImpl
{
    template<typename T_SpeciesType>
    struct apply
    {
        typedef DriftImpl< T_ParamClass, T_ValueFunctor > type;
    };

    HINLINE DriftImpl(uint32_t currentStep)
    {

    }

    template<typename T_Acc>
    struct Get
    {
        typedef detail::DriftImpl< T_ParamClass, T_ValueFunctor > type;
    };

    template<typename T_Acc>
    typename Get<T_Acc>::type
    DINLINE get(const T_Acc& acc, const DataSpace<simDim>& localCellIdx) const
    {
        typedef typename Get<T_Acc>::type Functor;

        return Functor();
    }
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
