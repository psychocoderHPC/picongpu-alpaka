/**
 * Copyright 2016 Rene Widera
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

#include "math/Vector.hpp"
#include "dimensions/DataSpace.hpp"

#include <type_traits>

namespace PMacc
{
template<typename T_Type, typename T_size, typename T_Sfinae = void>
struct Array;

template<typename T_Type, typename T_X, typename T_Y, typename T_Z>
struct Array<
    T_Type,
    PMacc::math::CT::Vector<T_X,T_Y,T_Z>,
    typename std::enable_if<
        PMacc::math::CT::Vector<T_X,T_Y,T_Z>::dim == 1
    >::type
> : ::cupla::Array<T_Type, PMacc::math::CT::Vector<T_X,T_Y,T_Z>::x::value>
{

    HDINLINE const T_Type&
    operator()( const DataSpace<DIM1>& idx ) const
    {
        return (*this)[idx.x()];
    }

    HDINLINE T_Type&
    operator()( const DataSpace<DIM1>& idx )
    {
        return (*this)[idx.x()];
    }
};

template<typename T_Type, typename T_X, typename T_Y, typename T_Z>
struct Array<
    T_Type,
    PMacc::math::CT::Vector<T_X,T_Y,T_Z>,
    typename std::enable_if<
        PMacc::math::CT::Vector<T_X,T_Y,T_Z>::dim == 2
    >::type
> : ::cupla::Array<
    ::cupla::Array<
        T_Type,
        PMacc::math::CT::Vector<T_X,T_Y,T_Z>::x::value
    >,
    PMacc::math::CT::Vector<T_X,T_Y,T_Z>::y::value
>
{
    HDINLINE Array ( const Array & ) = default;

    HDINLINE Array () = default;

    HDINLINE const T_Type&
    operator()( const DataSpace<DIM2>& idx ) const
    {
        return (*this)[idx.y()][idx.x()];
    }

    HDINLINE T_Type&
    operator()( const DataSpace<DIM2>& idx )
    {
        return (*this)[idx.y()][idx.x()];
    }
};

template<typename T_Type, typename T_X, typename T_Y, typename T_Z>
struct Array<
    T_Type,
    PMacc::math::CT::Vector<T_X,T_Y,T_Z>,
    typename std::enable_if<
        PMacc::math::CT::Vector<T_X,T_Y,T_Z>::dim == 3
    >::type
> : public ::cupla::Array<
        ::cupla::Array<
            ::cupla::Array<
                T_Type,
                PMacc::math::CT::Vector<T_X,T_Y,T_Z>::x::value
            >,
            PMacc::math::CT::Vector<T_X,T_Y,T_Z>::y::value
        >,
        PMacc::math::CT::Vector<T_X,T_Y,T_Z>::z::value
>
{

    HDINLINE Array ( const Array & ) = default;

    HDINLINE Array () = default;

    HDINLINE const T_Type&
    operator()( const DataSpace<DIM3>& idx ) const
    {
        return (*this)[idx.z()][idx.y()][idx.x()];
    }

    HDINLINE T_Type&
    operator()( const DataSpace<DIM3>& idx )
    {
        return (*this)[idx.z()][idx.y()][idx.x()];
    }
};

} //namespace PMacc
