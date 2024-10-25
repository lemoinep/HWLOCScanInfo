

project(main)

cmake_minimum_required(VERSION 3.22)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD 23)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD /02")
#string(REGEX REPLACE "/MD" "/02" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

#if($ENV{VERBOSE})
#    MESSAGE(STATUS "Benchmark GEMM")
#endif()


if(FALSE)
    find_package(MPI REQUIRED)
    if (MPI_FOUND)
        MESSAGE("{MPI_CXX_LIBRARIES}")
        else (MPI_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without MPI")
    endif(MPI_FOUND)
endif()



if(FALSE)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        MESSAGE("{OpenMP_CXX_LIBRARIES}")
        else (OpenMP_CXX_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without OpenMP")
    endif(OpenMP_CXX_FOUND)
endif()



if(FALSE)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message("${CMAKE_MODULE_PATH}")
    message("${EIGEN3_INCLUDE_DIR}")
endif()



if(FALSE)
    find_package(Threads)
endif()


if(FALSE)
    find_package(TBB REQUIRED)
    if (TBB_FOUND)
        MESSAGE("{TBB_CXX_LIBRARIES}")
    endif(TBB_FOUND)
endif()


if(FALSE)
    find_package(CUDA REQUIRED)
    if (CUDA_FOUND)
            MESSAGE("{CUDA_CXX_LIBRARIES}")
            MESSAGE(STATUS "Found headers CUDA : ${CUDA_INCLUDE_DIRS}")
            MESSAGE(STATUS "Found lib CUDA : ${CUDA_LIBRARIES}")
            MESSAGE(STATUS "Found CUDA nvcc : ${CUDA_NVCC_EXECUTABLE}")
            else (CUDA_FOUND)
            MESSAGE (SEND_ERROR "This application cannot compile without CUDA")
        endif(CUDA_FOUND)

    add_definitions(-D_FORCE_INLINES)
    #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21 -std=c++11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 )
endif()





if (TRUE)
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm-5.7.2/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

    find_package(HIP QUIET)

    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif()


    message(STATUS "HIP Path: " ${HIP_PATH})

    find_package(BLAS)
    find_package(LAPACK)
    find_package(hip)

    if($ENV{VERBOSE})
            MESSAGE(STATUS "GEMM -- BLAS_LIBRARIES   : ${BLAS_LIBRARIES}")
            MESSAGE(STATUS "GEMM -- BLAS_LINKER_FLAGS   : ${BLAS_LINKER_FLAGS}")
            MESSAGE(STATUS "GEMM -- LAPACK_LIBRARIES   : ${LAPACK_LIBRARIES}")
            MESSAGE(STATUS "GEMM -- LAPACK_LINKER_FLAGS   : ${LAPACK_LINKER_FLAGS}")
    endif()

    MESSAGE("")
    MESSAGE("{BEGIN::COMPILE_WITH_HIP}")

    include_directories(${HIP_PATH}/include)
    include_directories(/opt/rocm-5.7.2/hip/include)
    include_directories(/opt/rocm-5.7.2/hipblas/include)
    include_directories(/opt/rocm-5.7.2/hipblas/lib)

    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.7.2 /opt/rocm-5.7.2)

    # Set the GPU to compile for
    #set(GPU_TARGETS "gfx900:gfx906:gfx1032;gfx1035" CACHE STRING "GPU targets to compile for")

    set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
    set(GPU_RUNTIMES "HIP" "CUDA")
    set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

    if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
        set(ERROR_MESSAGE "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA.")
        message(FATAL_ERROR ${ERROR_MESSAGE})
    endif()

    enable_language(${GPU_RUNTIME})

    set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
    set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
    set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

    set(ROCM_ROOT "/opt/rocm-5.7.2" CACHE PATH "Root directory of the ROCm installation")
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

    MESSAGE("{END::COMPILE_WITH_HIP}")
    MESSAGE("")
endif()




if (TRUE)
    MESSAGE("{BEGIN::BUILD LINK WITH SPECX}")
    file(GLOB_RECURSE Specx_list "${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx_HIP/*.hpp")
    if (TRUE)
        foreach(file_path ${Specx_list})
            message(${file_path})
        endforeach()
    endif()

    MACRO(HEADER_DIRECTORIES return_list)
        #FILE(GLOB_RECURSE new_list ${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx/Src/*.hpp)
        FILE(GLOB_RECURSE new_list ${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx_HIP/Src/*.hpp)
        SET(dir_list "")
        FOREACH(file_path ${new_list})
            GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
            SET(dir_list ${dir_list} ${dir_path})
        ENDFOREACH()
        LIST(REMOVE_DUPLICATES dir_list)
        SET(${return_list} ${dir_list})
    ENDMACRO()

    HEADER_DIRECTORIES(header_specx_dir_list)

    list(LENGTH header_specx_dir_list header_dir_list_count)
    message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")

    if (TRUE)
        foreach(dir_path ${header_specx_dir_list})
            message(${dir_path})
        endforeach()
    endif()

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx_HIP/build/Src)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx_HIP/Src)


    MESSAGE("{END::COMPILE_WITH_SPECX}")
    MESSAGE("")
endif()




if (TRUE)
    MESSAGE("{BEGIN::BUILD LINK WITH HWLOC}")
    #find_package(hwloc REQUIRED)
    #target_link_library(main PRIVATE hwloc::hwloc)
    include(FindPkgConfig)
    if(PKG_CONFIG_EXECUTABLE)
    unset(HWLOC_FOUND CACHE)
    pkg_search_module(HWLOC hwloc)
    if(HWLOC_FOUND)
        message(STATUS "HWLOC_LIBRARIES=${HWLOC_LIBRARIES}")
        message(STATUS "HWLOC_LINK_LIBRARIES=${HWLOC_LINK_LIBRARIES}")
        message(STATUS "HWLOC_LIBRARY_DIRS=${HWLOC_LIBRARY_DIRS}")
        message(STATUS "HWLOC_LDFLAGS=${HWLOC_LDFLAGS}")
        message(STATUS "HWLOC_LDFLAGS_OTHERS=${HWLOC_LDFLAGS_OTHERS}")
        message(STATUS "HWLOC_INCLUDE_DIRS=${HWLOC_INCLUDE_DIRS}")
        message(STATUS "HWLOC_CFLAGS=${HWLOC_CFLAGS}")
        message(STATUS "HWLOC_CFLAGS_OTHER=${HWLOC_CFLAGS_OTHER}")
    else()
        message(FATAL_ERROR "HWLOC not found with pkg-config, add the path to hwloc.pc in PKG_CONFIG_PATH.")
    endif()
    else()
    message(FATAL_ERROR "PKG_CONFIG_EXECUTABLE: not found.")
    endif()
    MESSAGE("{END::BUILD LINK WITH HWLOC}")
    MESSAGE("")
endif()




if (TRUE)
    MESSAGE("{BEGIN::BUILD LINK WITH LIKWID}")

    include_directories(/home/u2/lemoine/likwid/src/includes) 
    include_directories(/home/u2/lemoine/likwid) 

    file(GLOB_RECURSE likwid_list
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../likwid/src/includes/*.h"
    )

    if (TRUE)
        foreach(file_path ${likwid_list})
            message(${file_path})
        endforeach()
    endif()


    MACRO(HEADER_DIRECTORIES return_list)
        FILE(GLOB_RECURSE new_list ${CMAKE_CURRENT_SOURCE_DIR}/../../../likwid/src/includes/*.h)
        SET(dir_list "")
        FOREACH(file_path ${new_list})
            GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
            SET(dir_list ${dir_list} ${dir_path})
        ENDFOREACH()
        LIST(REMOVE_DUPLICATES dir_list)
        SET(${return_list} ${dir_list})
    ENDMACRO()

    HEADER_DIRECTORIES(header_likwid_dir_list)

    list(LENGTH header_likwid_dir_list header_dir_list_count)
    message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")

    if (TRUE)
        foreach(dir_path ${header_likwid_dir_list})
            message(${dir_path})
        endforeach()
    endif()


    #   find_package(likwid REQUIRED)
    #   include_directories(${likwid_INCLUDE_DIRS})
    #   target_link_libraries(foo ${likwid_LIBRARIES})

    if (FALSE)
        include(FindPkgConfig)
        if(PKG_CONFIG_EXECUTABLE)
        unset(LIKWID_FOUND CACHE)
        pkg_search_module(LIKWID likwid)
        if(LIKWID_FOUND)
            message(STATUS "LIKWID_LIBRARIES=${LIKWID_LIBRARIES}")
            message(STATUS "LIKWID_LINK_LIBRARIES=${LIKWID_LINK_LIBRARIES}")
            message(STATUS "LIKWID_LIBRARY_DIRS=${LIKWID_LIBRARY_DIRS}")
            message(STATUS "LIKWID_LDFLAGS=${LIKWID_LDFLAGS}")
            message(STATUS "LIKWID_LDFLAGS_OTHERS=${LIKWID_LDFLAGS_OTHERS}")
            message(STATUS "LIKWID_INCLUDE_DIRS=${LIKWID_INCLUDE_DIRS}")
            message(STATUS "LIKWID_CFLAGS=${LIKWID_CFLAGS}")
            message(STATUS "LIKWID_CFLAGS_OTHER=${LIKWID_CFLAGS_OTHER}")
        else()
            message(FATAL_ERROR "LIKWID not found with pkg-config, add the path to likwid.pc in PKG_CONFIG_PATH.")
        endif()
        else()
        message(FATAL_ERROR "PKG_CONFIG_EXECUTABLE: not found.")
        endif()
    endif()
    MESSAGE("{END::BUILD LINK WITH LIKWID}")
    MESSAGE("")

endif()







message(CURRENT DIR= [${CMAKE_CURRENT_SOURCE_DIR}])

file(GLOB SOURCES
    "*.cpp"
)

add_executable(main ${SOURCES})


if(GPU_RUNTIME STREQUAL "HIP")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
    MESSAGE(${ROCM_ROOT}/include)
endif()

set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GPU_RUNTIME})

target_include_directories(main 
    PRIVATE 
        ${HWLOC_INCLUDE_DIRS}
    PRIVATE 
        ${header_likwid_dir_list}
    #PRIVATE
        #${header_specx_dir_list}
    PUBLIC
        ${MPI_CXX_INCLUDE_DIRS}
    #PUBLIC
        #OpenMP::OpenMP_CXX
    PRIVATE
        ${header_specx_dir_list}
)

target_compile_options(main PRIVATE ${HWLOC_CFLAGS})
target_link_libraries(main 
    PUBLIC 
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES} 
    PRIVATE 
        ${HWLOC_LINK_LIBRARIES}
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../likwid/liblikwid.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../likwid/liblikwid-hwloc.so.5.3
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../likwid/liblikwid-lua.so.5.3
    #PRIVATE 
        #hipblas
    #PUBLIC
        #${MPI_CXX_LIBRARIES}
    #PUBLIC
        #OpenMP::OpenMP_CXX
    #PUBLIC
        #${Eigen_LIBRARIES}
    #PUBLIC
        #TBB::tbb
    #PRIVATE
        #pthread
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx_HIP/build/Src/libspecx.a

)



#target_link_options(main PRIVATE ${HWLOC_LDFLAGS})
