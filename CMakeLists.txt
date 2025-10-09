cmake_minimum_required(VERSION 3.16...3.29)

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
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
)
FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
  FetchContent_Populate(glm)
  add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

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

  if(NOT DEFINED fetch_pybind_POPULATED)
    FetchContent_Declare(
      fetch_pybind
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v3.0.1 # v2.13.6
    )
    FetchContent_GetProperties(fetch_pybind)
    if(NOT fetch_pybind_POPULATED)
      FetchContent_Populate(fetch_pybind)
      add_subdirectory(
        ${fetch_pybind_SOURCE_DIR}
        ${fetch_pybind_BINARY_DIR}
      )
    endif()
  else()
    message(STATUS "Pybind11 was already fetched at: ${fetch_pybind_SOURCE_DIR}.")
  endif()
endif(RS_Build_PyBindings)

# Add the source files
file(GLOB SOURCES "src/*.cpp")

# Create the dynamic library
add_library(libRadFiled3D STATIC ${SOURCES})

# Add the include files
target_include_directories(libRadFiled3D PUBLIC include)
if (MSVC)
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

if(BUILD_TESTS MATCHES ON)
  if(NOT DEFINED fetch_googletest_POPULATED)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  enable_testing()
  include(GoogleTest)

	add_executable(basic_test tests/basic.cpp)
	if (MSVC)
    target_link_libraries(basic_test libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(basic_test libRadFiled3D stdc++fs GTest::gtest_main)
  endif()
	target_include_directories(basic_test PUBLIC include)
  set_target_properties(basic_test PROPERTIES FOLDER "Tests")

  add_executable(tracing_test tests/tracing_test.cpp)
	if (MSVC)
    target_link_libraries(tracing_test libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(tracing_test libRadFiled3D stdc++fs GTest::gtest_main)
  endif()
	target_include_directories(tracing_test PUBLIC include)
  set_target_properties(tracing_test PROPERTIES FOLDER "Tests")

  set_target_properties(gtest PROPERTIES FOLDER "GTest")
  set_target_properties(gmock PROPERTIES FOLDER "GTest")
  set_target_properties(gmock_main PROPERTIES FOLDER "GTest")
  set_target_properties(gtest_main PROPERTIES FOLDER "GTest")

  add_executable(MemoryTest ${PROJECT_SOURCE_DIR}/tests/memory_test.cpp)
  target_include_directories(MemoryTest PUBLIC include PRIVATE src)
  if (MSVC)
    target_link_libraries(MemoryTest glm libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(MemoryTest glm libRadFiled3D GTest::gtest_main stdc++fs)
  endif()
  set_property(TARGET MemoryTest PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  gtest_discover_tests(MemoryTest)

  set_target_properties(MemoryTest PROPERTIES FOLDER "Tests")

  add_executable(AccessorTest ${PROJECT_SOURCE_DIR}/tests/accessor_test.cpp)
  target_include_directories(AccessorTest PUBLIC include PRIVATE src)
  if (MSVC)
    target_link_libraries(AccessorTest glm libRadFiled3D GTest::gtest_main)
  else()
    target_link_libraries(AccessorTest glm libRadFiled3D GTest::gtest_main stdc++fs)
  endif()
  set_property(TARGET AccessorTest PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  gtest_discover_tests(AccessorTest)

  set_target_properties(AccessorTest PROPERTIES FOLDER "Tests")

  include(GoogleTest)
endif()


if(BUILD_EXAMPLES MATCHES ON)
  add_executable(example01 examples/cxx/example01.cpp)
  if (MSVC)
    target_link_libraries(example01 libRadFiled3D)
  else()
    target_link_libraries(example01 libRadFiled3D stdc++fs)
  endif()
  target_include_directories(example01 PUBLIC include)
  set_target_properties(example01 PROPERTIES FOLDER "Examples")
endif()

set_target_properties(libRadFiled3D PROPERTIES FOLDER "Library")

if(RS_Build_PyBindings)
  pybind11_add_module(RadFiled3D python/bindings/bindings.cpp)
  set_property(TARGET RadFiled3D PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
  add_dependencies(RadFiled3D libRadFiled3D)             
  if (MSVC)
    target_link_libraries(RadFiled3D PRIVATE libRadFiled3D)
  else()
    target_link_libraries(RadFiled3D PRIVATE libRadFiled3D stdc++fs)
  endif()
  set_target_properties(RadFiled3D PROPERTIES FOLDER "Library")
endif()