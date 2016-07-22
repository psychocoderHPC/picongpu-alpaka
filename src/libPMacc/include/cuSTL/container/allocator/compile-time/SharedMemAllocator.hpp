/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
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
#include "cuSTL/cursor/compile-time/BufferCursor.hpp"

namespace PMacc
{
namespace allocator
{
namespace CT
{
template<typename Type, typename Size, int dim = Size::dim, int uid = 0>
struct SharedMemAllocator;

template<typename Type, typename Size, int uid>
struct SharedMemAllocator<Type, Size, 1, uid>
{
    typedef Type type;
    typedef math::CT::UInt32<> Pitch;
    BOOST_STATIC_CONSTEXPR int dim = 1;
    typedef cursor::CT::BufferCursor<type, math::CT::UInt32<> > Cursor;

    template< typename T_Acc >
    DINLINE static Cursor allocate( const T_Acc& acc )
    {
        auto& shMem = ::alpaka::block::shared::st::allocVar<
            cupla::Array<Type,Size::x::value>,
            uid
        >(acc);
        return Cursor(&shMem[0]);
    }
};

template<typename Type, typename Size, int uid>
struct SharedMemAllocator<Type, Size, 2, uid>
{
    typedef Type type;
    typedef math::CT::UInt32<sizeof(Type) * Size::x::value> Pitch;
    BOOST_STATIC_CONSTEXPR int dim = 2;
    typedef cursor::CT::BufferCursor<type, Pitch> Cursor;

    template< typename T_Acc >
    DINLINE static Cursor allocate( const T_Acc& acc )
    {
        auto& shMem = ::alpaka::block::shared::st::allocVar<
            cupla::Array<cupla::Array<Type,Size::x::value>,Size::y::value>,
            uid
        >(acc);
        return Cursor(&shMem[0][0]);
    }
};

template<typename Type, typename Size, int uid>
struct SharedMemAllocator<Type, Size, 3, uid>
{
    typedef Type type;
    typedef math::CT::UInt32<sizeof(Type) * Size::x::value,
                             sizeof(Type) * Size::x::value * Size::y::value> Pitch;
    BOOST_STATIC_CONSTEXPR int dim = 3;
    typedef cursor::CT::BufferCursor<type, Pitch> Cursor;

    template< typename T_Acc >
    DINLINE static Cursor allocate( const T_Acc& acc )
    {
        auto& shMem = ::alpaka::block::shared::st::allocVar<
            cupla::Array<cupla::Array<cupla::Array<Type,Size::x::value>,Size::y::value>,Size::z::value>,
            uid
        >(acc);
        return Cursor(&shMem[0][0][0]);
    }
};

} // CT
} // allocator
} // PMacc

