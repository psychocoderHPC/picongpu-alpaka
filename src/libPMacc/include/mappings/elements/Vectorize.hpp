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

#include <type_traits>

namespace PMacc
{
namespace mappings
{
namespace elements
{



struct Contiguous
{
    template< typename T_IdxType >
    HDINLINE T_IdxType
    operator()( const T_IdxType& idx) const
    {
        return idx + 1;
    }
};



template< uint32_t T_dim, typename T_Functor, typename T_Size, typename T_Traverse, typename T_Sfinae = void  >
struct Vectorize;

template< typename T_Functor, typename T_Size, typename T_Traverse >
struct Vectorize<
    DIM1,
    T_Functor,
    T_Size,
    T_Traverse,
    typename std::enable_if<
        std::is_integral<
            T_Size
        >::value
    >::type

>
{
    HDINLINE void
    operator()( const T_Functor& functor, const T_Size& size, const T_Traverse& traverse ) const
    {
        using T_IdxType = T_Size;
        for( T_IdxType i = 0; i < size; i = traverse( i ))
            functor( i );
    }
};

template< typename T_Functor, typename T_Size>
struct Vectorize<
    DIM3,
    T_Functor,
    T_Size,
    Contiguous
>
{
    HDINLINE void
    operator()( const T_Functor& functor, const T_Size& size, const Contiguous& traverse ) const
    {
        using T_IdxType = typename T_Size::type;
        for( T_IdxType z = 0; z < size.z(); z = traverse( z ))
            for( T_IdxType y = 0; y < size.y(); y = traverse( y ))
                for( T_IdxType x = 0; x < size.x(); x = traverse( x ))
                    functor( T_Size(x,y,z) );
    }
};


template< uint32_t T_dim, typename T_Functor, typename T_Size, typename T_Traverse >
void vectorize( const T_Functor& functor, const T_Size& size, const T_Traverse& traverse )
{
    Vectorize< T_dim, T_Functor, T_Size, T_Traverse >()( functor, size, traverse);
}

} // namespace elements
} // namespace mappings
}// namespace PMacc
