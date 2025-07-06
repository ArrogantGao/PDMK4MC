#!/bin/bash

cmake -B build
cmake --build build -j 16
ctest --test-dir build