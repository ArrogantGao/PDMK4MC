cmake_minimum_required(VERSION 3.18)

project(hpdmk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_compile_options("-DSCTL_HAVE_SVML")
else()
  add_compile_options("-DSCTL_HAVE_LIBMVEC")
endif()

option(HPDMK_BUILD_TESTS "Build tests" ON)
option(HPDMK_BUILD_EXAMPLES "Build examples" ON)
option(HPDMK_INSTRUMENT "Instrument the code. Requires PAPI" OFF)

if (NOT ${HPDMK_BUILD_TESTS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDOCTEST_CONFIG_DISABLE")
  set(HPDMK_TEST_SRC)
else()
  set(HPDMK_TEST_SRC src/doctest.cpp)
endif()

add_compile_options("-march=native")
if(${HPDMK_INSTRUMENT})
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PAPI REQUIRED IMPORTED_TARGET papi)
  set(PAPI_TARGET papi)
  add_compile_options("-DSCTL_PROFILE=5;-DHPDMK_INSTRUMENT")
else()
  set(PAPI_TARGET "")
endif()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

set(HPDMK_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
  ${PROJECT_SOURCE_DIR}/extern/spdlog/include
  ${PROJECT_SOURCE_DIR}/extern/doctest
)

add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/nanobench EXCLUDE_FROM_ALL)
add_subdirectory(extern/doctest)

set(HPDMK_SRC src/hpdmk.cpp src/ewald.cpp
  ${HPDMK_TEST_SRC}
)
add_library(HPDMKOBJS OBJECT ${HPDMK_SRC})
set_target_properties(HPDMKOBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(HPDMKOBJS PUBLIC ${HPDMK_INCLUDES} ${MPI_CXX_INCLUDE_DIRS} ${PAPI_INCLUDEDIR})
target_link_libraries(HPDMKOBJS PUBLIC OpenMP::OpenMP_CXX ${PAPI_TARGET})
target_compile_options(HPDMKOBJS PUBLIC "-DSCTL_HAVE_MPI;-DSCTL_MAX_DEPTH=62;-funroll-loops")

add_library(hpdmk STATIC)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX HPDMKOBJS spdlog::spdlog)

add_library(hpdmk_shared SHARED)
set_target_properties(${PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_shared HPDMKOBJS OpenMP::OpenMP_CXX MPI::MPI_CXX BLAS::BLAS spdlog::spdlog doctest ${PAPI_TARGET})

if(${HPDMK_BUILD_TESTS})
  add_subdirectory(test)
endif()

if(${HPDMK_BUILD_EXAMPLES})
  add_subdirectory(examples)
endif()
