cmake_minimum_required(VERSION 3.18)
project(hpdmk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARK "Build benchmark" ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_compile_options("-DSCTL_HAVE_SVML")
else()
  add_compile_options("-DSCTL_HAVE_LIBMVEC")
endif()

add_compile_options("-march=native")

# add_compile_options("-DDEBUG")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

add_subdirectory(extern/vesin/vesin)

if(BUILD_TESTS)
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
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
)

add_library(OBJS OBJECT ${SRC})
set_target_properties(OBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(OBJS PUBLIC ${INCLUDES} ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(OBJS PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(OBJS PUBLIC "-DSCTL_HAVE_MPI;-DSCTL_MAX_DEPTH=62;-funroll-loops")

add_library(${PROJECT_NAME} STATIC)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX OBJS)

add_library(${PROJECT_NAME}_shared SHARED)
set_target_properties(${PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_shared OBJS OpenMP::OpenMP_CXX MPI::MPI_CXX BLAS::BLAS vesin)

if(BUILD_TESTS)
    enable_testing()
    set(TEST_SRC 
        # ${PROJECT_SOURCE_DIR}/test/hpdmk_test.cpp
        # ${PROJECT_SOURCE_DIR}/test/ewald_test.cpp
        ${PROJECT_SOURCE_DIR}/test/tree_test.cpp
    )

    add_executable(
        ${PROJECT_NAME}_test
        ${TEST_SRC}
    )
    target_link_libraries(
        ${PROJECT_NAME}_test
        GTest::gtest_main
        ${PROJECT_NAME}_shared
        vesin
    )

    # include(GoogleTest)
    # gtest_discover_tests(${PROJECT_NAME}_test)
endif()

if(BUILD_BENCHMARK)
  # add_compile_options("-pg")
  # add_link_options("-pg")

  add_executable(
      ${PROJECT_NAME}_benchmark
      ${PROJECT_SOURCE_DIR}/benchmark/dmk_runtime.cpp
  )
  target_link_libraries(
      ${PROJECT_NAME}_benchmark
      ${PROJECT_NAME}_shared
  )
endif()