# add_executable(test_ewald test_ewald.cpp)
# target_link_libraries(test_ewald hpdmk_shared doctest)

add_executable(test_hpdmk test_hpdmk.cpp)
target_link_libraries(test_hpdmk hpdmk_shared doctest)
