/**
 * Copyright 2015-2016 Rene Widera, Axel Huebl
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
struct NoneImpl
{
    HINLINE NoneImpl() = default;

    template<typename T_Particle1, typename T_Particle2, typename T_Acc>
    DINLINE void operator()(const T_Acc&,
                            T_Particle1&, T_Particle2&,
                            const bool, const bool)
    {
    }

};

} //namespace detail

struct NoneImpl
{
    template<typename T_SpeciesType>
    struct apply
    {
        typedef NoneImpl type;
    };

    HINLINE NoneImpl(const uint32_t)
    {
    }

    template<typename T_Acc>
    struct Get
    {
        typedef detail::NoneImpl type;
    };

    template<typename T_Acc>
    typename Get<T_Acc>::type
    DINLINE get(const T_Acc&, const DataSpace<simDim>& ) const
    {
        typedef typename Get<T_Acc>::type Functor;

        return Functor();
    }
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
