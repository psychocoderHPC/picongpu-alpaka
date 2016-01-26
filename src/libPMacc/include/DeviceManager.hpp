/**
 * Copyright 2015-2016 Benjamin Worpitz, Rene Widera
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

#include "types.h"

namespace PMacc
{

    /** Manage the alpaka devices
     *
     * holds one device and host accelerator.
     */
    class DeviceManager
    {
    public:
        /** Initialize the device manger
         *
         * @param uiIdx index of the device accelerator
         */
        void init(std::size_t uiIdx)
        {
            m_hostDev.reset(
                new alpaka::HostDev(
                    ::alpaka::dev::DevManCpu::getDevByIdx(
                        0u
                    )
                )
            );

            auto const uiNumDevices(
                ::alpaka::dev::DevMan< alpaka::AccDev >::getDevCount( )
            );

            // Beginning from the device given by the index, try if they are usable.
            for(
                std::size_t iDeviceOffset( 0 );
                iDeviceOffset < uiNumDevices;
                ++iDeviceOffset
            )
            {
                std::size_t const iDevice(
                    ( uiIdx + iDeviceOffset ) % uiNumDevices
                );

                try
                {
                    m_accDev.reset(
                        new alpaka::AccDev(
                            ::alpaka::dev::DevMan<alpaka::AccDev>::getDevByIdx(
                                iDevice
                            )
                        )
                    );
                    return;
                }
                catch( ... )
                { }
            }

            // If we came until here, none of the devices was usable.
            std::stringstream ssErr;
            ssErr << "Unable to return device handle for device " << uiIdx <<
                " because none of the " << uiNumDevices <<
                " devices is accessible!";
            throw std::runtime_error( ssErr.str( ) );
        }

        /** Get the selected device accelerator
         *
         * @return a device accelerator
         * @{
         */
        alpaka::AccDev const & getAccDevice( ) const
        {
            return *m_accDev;
        }

        alpaka::AccDev & getAccDevice( )
        {
            return *m_accDev;
        }
        //@}

        /** Get the selected host accelerator
         *
         * @return a host accelerator
         * @{
         */
        alpaka::HostDev const & getHostDevice( ) const
        {
            return *m_hostDev;
        }

        alpaka::HostDev & getHostDevice( )
        {
            return *m_hostDev;
        }
        //@}

        /** Get the instance of the DeviceManager
         *
         * @return instance of DeviceManager
         */
        static DeviceManager& getInstance( )
        {
            static DeviceManager instance;
            return instance;
        }

    private:
        /** storage of the device accelerator*/
        std::unique_ptr< alpaka::AccDev > m_accDev;
        /** storage of the host accelerator*/
        std::unique_ptr< alpaka::HostDev > m_hostDev;
    };

} //namespace PMacc
