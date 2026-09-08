cmake_minimum_required(VERSION 3.16)
project(adam_deploy_real)

set(CMAKE_BUILD_TYPE Release)  # Fixed typo
set(CMAKE_CXX_STANDARD 17)

# pndbotics SDK
set(PNDBOTICS_SDK_DIR /opt/pndbotics_sdk)
link_directories(${PNDBOTICS_SDK_DIR}/lib)
include_directories(${PNDBOTICS_SDK_DIR}/include)
include_directories(${PNDBOTICS_SDK_DIR}/include/ddscxx)
include_directories(${PNDBOTICS_SDK_DIR}/include/ddsc)
include_directories(/usr/local/include/iceoryx/v2.0.2)

# Torch setup
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/libtorch/share/cmake")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

# Add yaml-cpp subdirectory
set(YAML_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp)
add_subdirectory(${YAML_CPP_DIR})

# Link the libraries with correct order and paths
target_link_libraries(${PROJECT_NAME} pndbotics_sdk ddsc ddscxx ${TORCH_LIBRARIES}
    yaml-cpp::yaml-cpp
    pthread
)