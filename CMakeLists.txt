cmake_minimum_required(VERSION 3.18)
project(hpdmk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HPDMK_BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARK "Build benchmark" ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_compile_options("-DSCTL_HAVE_SVML")
else()
  add_compile_options("-DSCTL_HAVE_LIBMVEC")
endif()

# optimization flags
add_compile_options("-march=native")
add_compile_options("-ffast-math")
add_compile_options("-O3")

# add_compile_options("-DDEBUG")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_subdirectory(extern/vesin/vesin)

if(HPDMK_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
CPMAddPackage("gh:doctest/doctest@2.4.9")


set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(FINUFFT_BUILD_TESTS OFF CACHE BOOL "Whether to build the FINUFFT tests" FORCE)
CPMAddPackage("gh:flatironinstitute/finufft@2.4.1")

set(INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
  ${PROJECT_SOURCE_DIR}/extern/vesin/vesin/include
)

set(SRC 
    ${PROJECT_SOURCE_DIR}/src/hpdmk.cpp
    ${PROJECT_SOURCE_DIR}/src/ewald.cpp
    ${PROJECT_SOURCE_DIR}/src/tree.cpp
    ${PROJECT_SOURCE_DIR}/src/planewave.cpp
    ${PROJECT_SOURCE_DIR}/src/energy.cpp
    ${PROJECT_SOURCE_DIR}/src/energy_direct.cpp
    # ${PROJECT_SOURCE_DIR}/src/energy_shift.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/pswf.cpp
)

add_library(OBJS OBJECT ${SRC})
set_target_properties(OBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(OBJS PUBLIC ${INCLUDES} ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(OBJS PUBLIC OpenMP::OpenMP_CXX BLAS::BLAS LAPACK::LAPACK finufft)
target_compile_options(OBJS PUBLIC "-DSCTL_HAVE_MPI;-DSCTL_MAX_DEPTH=62;-funroll-loops")

# compile the static library
# add_library(${PROJECT_NAME} STATIC)
# target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX BLAS::BLAS LAPACK::LAPACK OBJS finufft)

# compile the shared library
add_library(${PROJECT_NAME}_shared SHARED)
set_target_properties(${PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_shared OBJS OpenMP::OpenMP_CXX MPI::MPI_CXX BLAS::BLAS LAPACK::LAPACK vesin finufft)

add_executable(${PROJECT_NAME}_energy ${PROJECT_SOURCE_DIR}/example/dmk_energy.cpp)
target_link_libraries(${PROJECT_NAME}_energy ${PROJECT_NAME}_shared)

if(HPDMK_BUILD_TESTS)
    enable_testing()
    set(TEST_SRC 
        ${PROJECT_SOURCE_DIR}/test/ewald_test.cpp
        ${PROJECT_SOURCE_DIR}/test/utils_test.cpp
        ${PROJECT_SOURCE_DIR}/test/pswf_test.cpp
        ${PROJECT_SOURCE_DIR}/test/nudft_test.cpp
        ${PROJECT_SOURCE_DIR}/test/hpdmk_test.cpp
    )

    add_executable(
        ${PROJECT_NAME}_test
        ${TEST_SRC}
    )
    target_link_libraries(
        ${PROJECT_NAME}_test
        GTest::gtest_main
        ${PROJECT_NAME}_shared
    )

    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME}_test)
endif()

# if(BUILD_BENCHMARK)
#     # add_executable(
#     #     ${PROJECT_NAME}_energy_runtime
#     #     ${PROJECT_SOURCE_DIR}/benchmark/dmk_energy_runtime.cpp
#     # )
#     # target_link_libraries(${PROJECT_NAME}_energy_runtime ${PROJECT_NAME}_shared)

#     # add_executable(
#     #     ${PROJECT_NAME}_potential_accuracy
#     #     ${PROJECT_SOURCE_DIR}/benchmark/potential_accuracy.cpp
#     # )
#     # target_link_libraries(${PROJECT_NAME}_potential_accuracy ${PROJECT_NAME}_shared)

#     # add_executable(
#     #     ${PROJECT_NAME}_ewald_potential_runtime
#     #     ${PROJECT_SOURCE_DIR}/benchmark/ewald_potential_runtime.cpp
#     # )
#     # target_link_libraries(${PROJECT_NAME}_ewald_potential_runtime ${PROJECT_NAME}_shared)

#     # add_executable(
#     #     ${PROJECT_NAME}_dmk_potential_runtime
#     #     ${PROJECT_SOURCE_DIR}/benchmark/dmk_potential_runtime.cpp
#     # )
#     # target_link_libraries(${PROJECT_NAME}_dmk_potential_runtime ${PROJECT_NAME}_shared)

#     add_executable(
#         ${PROJECT_NAME}_shift_runtime
#         ${PROJECT_SOURCE_DIR}/benchmark/dmk_shift_runtime.cpp
#     )
#     target_link_libraries(${PROJECT_NAME}_shift_runtime ${PROJECT_NAME}_shared)
# endif()