# MIT License 
# Copyright (c) 2018-Today Michele Adduci <michele@adduci.org>
#
# Project-related instructions

# Activate Testing, if given
if(BUILD_TESTING)
  enable_testing()
endif()

#####################################
# Define Targets
#####################################
add_subdirectory(ewald)

#####################################
# Define Install Targets
#####################################
install(TARGETS
  ewald
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)

install(
  FILES 
    ${CMAKE_SOURCE_DIR}/project/ewald/include/ewald.h 
  DESTINATION 
    include/ewald)

install(FILES  
  ${CMAKE_SOURCE_DIR}/LICENSE
DESTINATION .)