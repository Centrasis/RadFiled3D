cmake_minimum_required(VERSION 3.16...4.2)

# Set the project name
project(RadiationFieldStorage LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

option(BUILD_TESTS "Build the tests" OFF)
option(BUILD_EXAMPLES "Build the examples" OFF)
option(RS_Build_PyBindings "If the python bindings should be build" ON)

message(STATUS "Download GLM...")
# GLM ships its own tests and install rules; both are turned off so that a
# `cmake --install` (as run by scikit-build-core when assembling the wheel)
# only ever contains RadFiled3D's own artifacts.
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.3
  GIT_SHALLOW    TRUE
)
# FetchContent_MakeAvailable instead of the deprecated single-argument
# FetchContent_Populate(): the latter is an error under CMake >= 3.30 policy
# CMP0169, which any CMake 4.x provisioned by pip would enforce.
FetchContent_MakeAvailable(glm)

if(RS_Build_PyBindings)
  set(PYTHON_VERSION "3.11" CACHE STRING "Python Version to use for pyBind11 and conda.")
  set_property(CACHE PYTHON_VERSION PROPERTY STRINGS  "3.14" "3.13" "3.12" "3.11" "3.10")
  set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})

  if(NOT Python_FOUND)
    set(DESIRED_PYTHON_VERSION ${PYTHON_VERSION})
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    
    if("${Python_VERSION}" MATCHES ".*${DESIRED_PYTHON_VERSION}.*")
      message(STATUS "Found suitable Python version: ${Python_VERSION}")
    else()
      message(WARNING "Desired Python version ${DESIRED_PYTHON_VERSION} not found. Found version: ${Python_VERSION}")
    endif()
    
    set(PYTHON_LIBRARY_DIR "${Python_SITELIB}")
    message(STATUS "Found python site package: ${PYTHON_LIBRARY_DIR}")
  else()
    message(STATUS "Python was already found: ${Python_EXECUTABLE}")
  endif()

  # Prefer a pybind11 provided by the build environment (scikit-build-core adds
  # the build-requirement's CMake config to CMAKE_PREFIX_PATH). Fall back to
  # fetching it for standalone CMake builds without pybind11 installed.
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    message(STATUS "Found pybind11 via find_package: ${pybind11_VERSION}")
  else()
    message(STATUS "pybind11 not found, fetching via FetchContent...")
    FetchContent_Declare(
      fetch_pybind
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v3.1.0
      GIT_SHALLOW    TRUE
    )
    # See the GLM comment above: MakeAvailable is the CMake >= 3.30 safe form.
    FetchContent_MakeAvailable(fetch_pybind)
  endif()
endif(RS_Build_PyBindings)

# Add the source files
file(GLOB SOURCES "src/*.cpp")

# Create the dynamic library
add_library(libRadFiled3D STATIC ${SOURCES})

# Add the include files
target_include_directories(libRadFiled3D PUBLIC include)
if (MSVC OR APPLE)
  target_link_libraries(libRadFiled3D PUBLIC glm)
else()
  target_link_libraries(libRadFiled3D PUBLIC glm stdc++fs)
endif()


set_target_properties(libRadFiled3D
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

if(RS_Build_PyBindings)
  pybind11_add_module(RadFiled3D python/bindings/bindings.cpp)
  set_property(TARGET RadFiled3D PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  add_dependencies(RadFiled3D libRadFiled3D)             
  if (MSVC OR APPLE)
    target_link_libraries(RadFiled3D PRIVATE libRadFiled3D)
  else()
    target_link_libraries(RadFiled3D PRIVATE libRadFiled3D stdc++fs)
  endif()
  set_target_properties(RadFiled3D PROPERTIES FOLDER "Library")

  # Install the native module into the RadFiled3D package (used by
  # scikit-build-core when assembling the wheel). The pure-python files and the
  # RadFiled3D.pyi stub are added by scikit-build-core from python/RadFiled3D.
  install(TARGETS RadFiled3D
    LIBRARY DESTINATION RadFiled3D
    RUNTIME DESTINATION RadFiled3D
  )
endif()

if(BUILD_TESTS MATCHES ON)
  if(NOT TARGET GTest::gtest_main)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  enable_testing()
  include(GoogleTest)

	add_executable(basic_test tests/basic.cpp)
	if (MSVC OR APPLE)
    target_link_libraries(basic_test libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(basic_test libRadFiled3D stdc++fs GTest::gtest_main)
  endif()
	target_include_directories(basic_test PUBLIC include)
  set_target_properties(basic_test PROPERTIES FOLDER "Tests")
  gtest_discover_tests(basic_test)

  add_executable(tracing_test tests/tracing_test.cpp)
	if (MSVC OR APPLE)
    target_link_libraries(tracing_test libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(tracing_test libRadFiled3D stdc++fs GTest::gtest_main)
  endif()	
	target_include_directories(tracing_test PUBLIC include)
  set_target_properties(tracing_test PROPERTIES FOLDER "Tests")
  gtest_discover_tests(tracing_test)

  set_target_properties(gtest PROPERTIES FOLDER "GTest")
  set_target_properties(gmock PROPERTIES FOLDER "GTest")
  set_target_properties(gmock_main PROPERTIES FOLDER "GTest")
  set_target_properties(gtest_main PROPERTIES FOLDER "GTest")

  add_executable(MemoryTest ${PROJECT_SOURCE_DIR}/tests/memory_test.cpp)
  target_include_directories(MemoryTest PUBLIC include PRIVATE src)
  if (MSVC OR APPLE)
    target_link_libraries(MemoryTest glm libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(MemoryTest glm libRadFiled3D GTest::gtest_main stdc++fs)
  endif()
  set_property(TARGET MemoryTest PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  gtest_discover_tests(MemoryTest)

  set_target_properties(MemoryTest PROPERTIES FOLDER "Tests")

  add_executable(AccessorTest ${PROJECT_SOURCE_DIR}/tests/accessor_test.cpp)
  target_include_directories(AccessorTest PUBLIC include PRIVATE src)
  if (MSVC OR APPLE)
    target_link_libraries(AccessorTest glm libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(AccessorTest glm libRadFiled3D GTest::gtest_main stdc++fs)
  endif()
  set_property(TARGET AccessorTest PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  gtest_discover_tests(AccessorTest)

  set_target_properties(AccessorTest PROPERTIES FOLDER "Tests")

  add_executable(SphericalTest ${PROJECT_SOURCE_DIR}/tests/spherical_test.cpp)
  target_include_directories(SphericalTest PUBLIC include PRIVATE src)
  if (MSVC OR APPLE)
    target_link_libraries(SphericalTest glm libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(SphericalTest glm libRadFiled3D GTest::gtest_main stdc++fs)
  endif()
  set_property(TARGET SphericalTest PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  gtest_discover_tests(SphericalTest)
  set_target_properties(SphericalTest PROPERTIES FOLDER "Tests")

  add_executable(VoxelCountTest ${PROJECT_SOURCE_DIR}/tests/voxel_count_test.cpp)
  target_include_directories(VoxelCountTest PUBLIC include)
  if (MSVC OR APPLE)
    target_link_libraries(VoxelCountTest glm libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(VoxelCountTest glm libRadFiled3D GTest::gtest_main stdc++fs)
  endif()
  set_property(TARGET VoxelCountTest PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  gtest_discover_tests(VoxelCountTest)
  set_target_properties(VoxelCountTest PROPERTIES FOLDER "Tests")

  include(GoogleTest)
endif()


if(BUILD_EXAMPLES MATCHES ON)
  add_executable(example01 examples/cxx/example01.cpp)
  if (MSVC OR APPLE)
    target_link_libraries(example01 libRadFiled3D)
  else()
    target_link_libraries(example01 libRadFiled3D stdc++fs)
  endif()
  target_include_directories(example01 PUBLIC include)
  set_target_properties(example01 PROPERTIES FOLDER "Examples")
endif()

set_target_properties(libRadFiled3D PROPERTIES FOLDER "Library")
