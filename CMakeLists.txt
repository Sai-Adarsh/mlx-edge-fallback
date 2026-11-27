cmake_minimum_required(VERSION 3.27)
project(mlx_edge_fallback LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(EDGE_ENABLE_MLX "Build the Apple Silicon local model backend" OFF)
set(TOKENIZERS_CPP_SOURCE "" CACHE PATH "Path to a recursive tokenizers-cpp checkout")

find_package(CURL REQUIRED)
find_package(nlohmann_json 3.12 CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_executable(edge_fallback src/main.cpp)
target_link_libraries(edge_fallback PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)

if(EDGE_ENABLE_MLX)
  if(NOT TOKENIZERS_CPP_SOURCE)
    message(FATAL_ERROR "Set TOKENIZERS_CPP_SOURCE to a recursive tokenizers-cpp checkout.")
  endif()
  find_package(MLX CONFIG REQUIRED)
  add_subdirectory("${TOKENIZERS_CPP_SOURCE}" tokenizers-cpp EXCLUDE_FROM_ALL)
  target_sources(edge_fallback PRIVATE src/mlx_model.cpp)
  target_compile_definitions(edge_fallback PRIVATE EDGE_HAS_MLX=1)
  target_link_libraries(edge_fallback PRIVATE mlx tokenizers_cpp)
endif()
