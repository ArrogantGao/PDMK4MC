# MIT License
# ewald target instructions

#####################################
# Define Target Information
#####################################
set(TARGET_NAME ewald)
set(TARGET_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/ewald.cpp)

#####################################
# SCTL Configuration
#####################################
# Set SCTL include directory
set(SCTL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/SCTL/include)

# Find required dependencies for SCTL
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(PkgConfig REQUIRED)

# Find FFTW3 libraries
pkg_check_modules(FFTW3 REQUIRED fftw3)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(FFTW3L REQUIRED fftw3l)
# pkg_check_modules(FFTW3_OMP REQUIRED fftw3_omp)

#####################################
# Support IDE Visualization
#####################################
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TARGET_SOURCES})

#####################################
# Define Target according to options
#####################################
if(BUILD_SHARED_LIBS)
  add_library(${TARGET_NAME} SHARED ${TARGET_SOURCES})
else()
  add_library(${TARGET_NAME} STATIC ${TARGET_SOURCES})
endif()

#####################################
# Define Target Properties
#####################################

# generate an export header to expose symbols when using the library
generate_export_header(${TARGET_NAME} EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/exports/${TARGET_NAME}_export.h)

# add target properties such as debug postfix, MSVC Runtime and library version
set_target_properties(${TARGET_NAME} PROPERTIES 
                                      DEBUG_POSTFIX "-d" 
                                      MSVC_RUNTIME_LIBRARY "${MSVC_RUNTIME_TYPE}" 
                                      VERSION ${CMAKE_PROJECT_VERSION}
                                      SOVERSION ${CMAKE_PROJECT_VERSION})

# set target compile options as defined in the cmake/compiler_options.cmake Module
target_compile_options(${TARGET_NAME} PRIVATE ${compiler_options})

# add compiler definitions
target_compile_definitions(${TARGET_NAME} PRIVATE ${compiler_definitions})

# Add SCTL-specific compiler definitions
target_compile_definitions(${TARGET_NAME} PRIVATE 
    SCTL_GLOBAL_MEM_BUFF=0
    SCTL_PROFILE=5
    SCTL_VERBOSE
    SCTL_SIG_HANDLER
    SCTL_QUAD_T=__float128
    SCTL_HAVE_BLAS
    SCTL_HAVE_LAPACK
    SCTL_FFTW_THREADS
    SCTL_HAVE_FFTW
    SCTL_HAVE_FFTWF
    # SCTL_HAVE_FFTWL
)

# set target link options as defined in the cmake/compiler_options.cmake Module
target_link_options(${TARGET_NAME} PRIVATE ${linker_flags})

# include all the necessary directories
target_include_directories(${TARGET_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
  PRIVATE 
    ${TARGET_INCLUDE_FOLDER}
    ${SCTL_INCLUDE_DIR}
    # Any other extra include required
)

# Link SCTL dependencies
target_link_libraries(${TARGET_NAME} PUBLIC 
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${FFTW3_LIBRARIES}
    ${FFTW3F_LIBRARIES}
    ${FFTW3L_LIBRARIES}
    # ${FFTW3_OMP_LIBRARIES}
)

if(BUILD_TESTING)
  list(APPEND TEST_CASES
    ewald_test
  )
  foreach(TEST_CASE ${TEST_CASES})
    add_executable(${TEST_CASE} ${CMAKE_CURRENT_SOURCE_DIR}/test/${TEST_CASE}.cpp)
    target_link_libraries(${TEST_CASE} PRIVATE ${TARGET_NAME} doctest::doctest)
    target_compile_options(${TEST_CASE} PRIVATE ${compiler_options})
    target_compile_definitions(${TEST_CASE} PRIVATE ${compiler_definitions})
    target_link_options(${TEST_CASE} PRIVATE ${linker_flags})

    target_include_directories(${TEST_CASE}
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
      PRIVATE 
        ${TARGET_INCLUDE_FOLDER}
        ${SCTL_INCLUDE_DIR}
        # Any other extra include required
    )

    # Link SCTL dependencies for the test executable as well
    target_link_libraries(${TEST_CASE} PRIVATE 
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${FFTW3_LIBRARIES}
        ${FFTW3F_LIBRARIES}
        ${FFTW3L_LIBRARIES}
        # ${FFTW3_OMP_LIBRARIES}
    )

    add_test(NAME ${TEST_CASE} COMMAND ${TEST_CASE})
  endforeach(TEST_CASE ${TEST_CASES})
endif()