# Hybird Periodic Dual-space multilevel kernel-splitting method

## Working in progress

- [x] Package set up (following the dmk project by dmalhotra)
- [ ] Ewald solver
- [ ] Planewave version of dmk solver
- [ ] Hybrid version of dmk solver

## Compile and run
```bash
git clone https://github.com/ArrogantGao/HybridPeriodicDMK
cd HybridPeriodicDMK
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DHPDMK_BUILD_TESTS=ON -DHPDMK_BUILD_EXAMPLES=OFF
make -j 16
```

The test can be run by
```bash
cd build
cd test
./test_hpdmk
./test_ewald
```