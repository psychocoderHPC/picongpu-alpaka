# Copyright 2013-2015 Rene Widera, Axel Huebl, Benjamin Worpitz
#
# This file is part of libPMacc.
#
# libPMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libPMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with libPMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.3)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project
################################################################################

project(gameOfLife)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF()

# Set helper paths to find libraries and packages.
LIST(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../../thirdParty/cmake-modules")

################################################################################
# Configure Dependencies
################################################################################

#-------------------------------------------------------------------------------
# Find Boost.
#-------------------------------------------------------------------------------
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex system filesystem)
LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${Boost_INCLUDE_DIRS})
LIST(APPEND _GOL_LIBRARIES_PRIVATE ${Boost_LIBRARIES})

#-------------------------------------------------------------------------------
# Find PMacc.
#-------------------------------------------------------------------------------
SET("PMACC_ROOT" "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE STRING  "The location of the PMacc library")
LIST(APPEND CMAKE_MODULE_PATH "${PMACC_ROOT}")

FIND_PACKAGE(PMacc REQUIRED)
LIST(APPEND _GOL_DEFINITIONS_PRIVATE ${PMacc_DEFINITIONS})
LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${PMacc_INCLUDE_DIRS})
LIST(APPEND _GOL_LIBRARIES_PRIVATE ${PMacc_LIBRARIES})

#-------------------------------------------------------------------------------
# Find PNGwriter
#-------------------------------------------------------------------------------
find_package(PNGwriter 0.5.6)
IF(PNGwriter_FOUND)
    LIST(APPEND _GOL_DEFINITIONS_PRIVATE ${PNGwriter_DEFINITIONS} "-DGOL_IMG_PNG=1")
    LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${PNGwriter_INCLUDE_DIRS})
    LIST(APPEND _GOL_LIBRARIES_PRIVATE ${PNGwriter_LIBRARIES})
ENDIF(PNGwriter_FOUND)

#-------------------------------------------------------------------------------
# VampirTrace
#-------------------------------------------------------------------------------
OPTION(VAMPIR_ENABLE "Create PIConGPU with VampirTrace support" OFF)

# SET filters: please do NOT use line breaks WITHIN the string!
SET(VT_INST_FILE_FILTER
    "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBufferIntern.hpp,DeviceBuffer.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBufferIntern.hpp,HostBuffer.hpp"
    CACHE STRING "VampirTrace: Files to exclude from instrumentation")
SET(VT_INST_FUNC_FILTER
    "vector,Vector,dim3,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,CudaEvent,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance"
    CACHE STRING "VampirTrace: Functions to exclude from instrumentation")

IF(VAMPIR_ENABLE)
    MESSAGE(STATUS "Building with VampirTrace support")
    SET(VAMPIR_ROOT "$ENV{VT_ROOT}")
    IF(NOT VAMPIR_ROOT)
        MESSAGE(FATAL_ERROR "Environment variable VT_ROOT not SET!")
    ENDIF()

    # compile flags
    EXECUTE_PROCESS(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-compile
                    OUTPUT_VARIABLE VT_COMPILEFLAGS
                    RESULT_VARIABLE VT_CONFIG_RETURN
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF(NOT VT_CONFIG_RETURN EQUAL 0)
        MESSAGE(FATAL_ERROR "Can NOT execute 'vtc++' at $ENV{VT_ROOT}/bin/vtc++ - check FILE permissions")
    ENDIF()
    # link flags
    EXECUTE_PROCESS(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-link
                    OUTPUT_VARIABLE VT_LINKFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    # bugfix showme
    string(REPLACE "--as-needed" "--no-as-needed" VT_LINKFLAGS "${VT_LINKFLAGS}")

    # modify our flags
    LIST(APPEND CMAKE_CXX_LINK_FLAGS ${VT_LINKFLAGS})
    LIST(APPEND CMAKE_CXX_FLAGS ${VT_COMPILEFLAGS})
    LIST(APPEND CMAKE_CXX_FLAGS "-finstrument-functions-exclude-FILE-list=${VT_INST_FILE_FILTER}")
    LIST(APPEND CMAKE_CXX_FLAGS "-finstrument-functions-exclude-function-list=${VT_INST_FUNC_FILTER}")

    # nvcc flags (rly necessary?)
    LIST(APPEND CUDA_NVCC_FLAGS
        -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-FILE-list=\\\"${VT_INST_FILE_FILTER}\\\"
        -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${VT_INST_FUNC_FILTER}\\\"
        -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${VT_ROOT}/include/vampirtrace\\\"
        -v)

    # for manual instrumentation and hints that vampir is enabled in our code
    LIST(APPEND _GOL_DEFINITIONS_PRIVATE VTRACE)

    # titan work around: currently (5.14.4) the -D defines are not provided by -vt:showme-compile
    LIST(APPEND _GOL_DEFINITIONS_PRIVATE MPICH_IGNORE_CXX_SEEK)
ENDIF(VAMPIR_ENABLE)

################################################################################
# Compile and link.
################################################################################
# Add all the include files in all recursive subdirectories and group them accordingly for MSVC projects.
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}/include" "${CMAKE_CURRENT_LIST_DIR}" "hpp" _GOL_FILES_HEADER)
append_recursive_files_add_to_src_group("${CMAKE_CURRENT_LIST_DIR}/" "${CMAKE_CURRENT_LIST_DIR}" "cpp" _GOL_FILES_SOURCE)

LIST(APPEND _GOL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
LIST(APPEND _GOL_INCLUDE_DIRECTORIES_PRIVATE ${_GOL_INCLUDE_DIR})

ADD_DEFINITIONS(${_GOL_DEFINITIONS_PRIVATE})
INCLUDE_DIRECTORIES(${_GOL_INCLUDE_DIRECTORIES_PRIVATE})

# Always add all files to the target executable build call to add them to the build project.
ALPAKA_ADD_EXECUTABLE(
    "gameOfLife"
    ${_GOL_FILES_HEADER}
    ${_GOL_FILES_SOURCE})
# Set the link libraries for this library (adds libs, include directories, defines and compile options).
TARGET_LINK_LIBRARIES(
    "gameOfLife"
    PUBLIC "alpaka")

SET_TARGET_PROPERTIES(${_PROJECT_NAME} PROPERTIES FOLDER "example")

################################################################################
# Install
################################################################################

SET(GOL_COPY_ON_INSTALL "submit" CACHE LIST "folder which copied on install to install path" )

INSTALL(
    TARGETS gameOfLife
    RUNTIME DESTINATION bin)

#copy all subfolders (defined in: GOL_COPY_ON_INSTALL) to install folder
FOREACH(dir ${GOL_COPY_ON_INSTALL})
  #if source not exists than copy
  IF(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
      #copy important sub-folders from extension path (default PIConGPU parameter)
      IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
        INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
          DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
          PATTERN .svn EXCLUDE
        )
      ENDIF()
  ENDIF()
ENDFOREACH()
