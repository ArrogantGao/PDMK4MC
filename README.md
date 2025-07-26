# Hybrid periodic dual-space multilevel kernel-splitting method for long-range electrostatics

This is a C++ implementation of the hybrid periodic dual-space multilevel kernel-splitting method for long-range electrostatics.

## Work to do

- [x] set up cpp project and ci
- [x] implement the ewald summation
- [x] implement the planewave version of hpdmk
- [ ] detailed unit tests

## Next steps

- [ ] add different kernels other than Gaussian
- [ ] implement the chebyshev version of hpdmk

- [ ] user level interface (not necessary for now)

## How to use

1. Clone the repository
2. Run `cmake -B build` to create the build directory
3. Run `cmake --build build -j 8` to build the project with 8 threads
4. Run `ctest --test-dir build` to run the tests
