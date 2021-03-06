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

#include "pmacc_types.hpp"

namespace PMacc
{
namespace nvidia
{
namespace rng
{
namespace methods
{

template< typename T_Acc >
class Xor
{
private:
     using Gen =
        decltype(
            ::alpaka::rand::generator::createDefault(
                std::declval<T_Acc const &>(),
                std::declval<uint32_t &>(),
                std::declval<uint32_t &>()));
    PMACC_ALIGN(gen, Gen);
public:
    typedef Gen StateType;
    typedef T_Acc Acc;

    HDINLINE Xor() : gen (0)
    {
    }

    DINLINE Xor(const T_Acc& acc, uint32_t seed, uint32_t subsequence = 0)
    {
        gen = ::alpaka::rand::generator::createDefault(acc, seed, subsequence);
    }

    HDINLINE Xor(const Xor& other): gen(other.gen)
    {

    }

protected:

    DINLINE StateType& getState()
    {
        return gen;
    }
};
}
}
}
}
