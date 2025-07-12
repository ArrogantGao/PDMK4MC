# Hybrid periodic dual-space multilevel kernel-splitting method for long-range electrostatics

This is a C++ implementation of the hybrid periodic dual-space multilevel kernel-splitting method for long-range electrostatics.

## Work to do

- [x] set up cpp project and ci
- [x] implement the ewald summation
- [ ] implement the planewave version of hpdmk
- [ ] implement the chebyshev version of hpdmk

## How to use

1. Clone the repository
2. Run `cmake -B build` to create the build directory
3. Run `cmake --build build` to build the project
4. Run `ctest --test-dir build` to run the tests
