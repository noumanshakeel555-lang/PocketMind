cmake_minimum_required(VERSION 3.22.1)
project(llamachat)

# -----------------------------------------------------------------------------
# Android Studio / Ninja setup
# -----------------------------------------------------------------------------

# Get Android Studio's bundled CMake/Ninja folder
get_filename_component(ANDROID_CMAKE_BIN_DIR "${CMAKE_COMMAND}" DIRECTORY)

# Use HOST check because we are building on Windows but targeting Android
if(CMAKE_HOST_WIN32)
    set(ANDROID_NINJA_PATH "${ANDROID_CMAKE_BIN_DIR}/ninja.exe")
    set(ENV{PATH} "${ANDROID_CMAKE_BIN_DIR};$ENV{PATH}")
else()
    set(ANDROID_NINJA_PATH "${ANDROID_CMAKE_BIN_DIR}/ninja")
    set(ENV{PATH} "${ANDROID_CMAKE_BIN_DIR}:$ENV{PATH}")
endif()

if(EXISTS "${ANDROID_NINJA_PATH}")
    set(CMAKE_MAKE_PROGRAM "${ANDROID_NINJA_PATH}" CACHE FILEPATH "Ninja path" FORCE)
    message(STATUS "Using Ninja: ${CMAKE_MAKE_PROGRAM}")
    message(STATUS "Added CMake bin to PATH: ${ANDROID_CMAKE_BIN_DIR}")
else()
    message(FATAL_ERROR "Ninja not found at: ${ANDROID_NINJA_PATH}")
endif()

# -----------------------------------------------------------------------------
# Vulkan SDK headers
#
# Android NDK gives us libvulkan.so and glslc, but llama.cpp Vulkan also checks
# for the C++ header vulkan/vulkan.hpp. That header usually comes from the
# LunarG Vulkan SDK on Windows.
# -----------------------------------------------------------------------------

if(DEFINED ENV{VULKAN_SDK})
    set(HOST_VULKAN_SDK "$ENV{VULKAN_SDK}")
    set(HOST_VULKAN_INCLUDE_DIR "${HOST_VULKAN_SDK}/Include")

    message(STATUS "Using host Vulkan SDK: ${HOST_VULKAN_SDK}")

    if(EXISTS "${HOST_VULKAN_INCLUDE_DIR}/vulkan/vulkan.hpp")
        message(STATUS "Found Vulkan-Hpp header: ${HOST_VULKAN_INCLUDE_DIR}/vulkan/vulkan.hpp")

        # Make the header visible globally before adding llama.cpp.
        include_directories("${HOST_VULKAN_INCLUDE_DIR}")

        # Also force it into C/C++ flags so subprojects such as llama.cpp can see it.
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${HOST_VULKAN_INCLUDE_DIR}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HOST_VULKAN_INCLUDE_DIR}")

        # Helpful hints for find_package/find_path based checks.
        set(Vulkan_INCLUDE_DIR "${HOST_VULKAN_INCLUDE_DIR}" CACHE PATH "Vulkan include directory" FORCE)
        set(VULKAN_HPP_INCLUDE_DIR "${HOST_VULKAN_INCLUDE_DIR}" CACHE PATH "Vulkan-Hpp include directory" FORCE)
    else()
        message(WARNING "VULKAN_SDK is set, but vulkan/vulkan.hpp was not found at: ${HOST_VULKAN_INCLUDE_DIR}/vulkan/vulkan.hpp")
    endif()
else()
    message(WARNING "VULKAN_SDK environment variable is not set. llama.cpp Vulkan may fail if vulkan/vulkan.hpp is missing.")
endif()

# -----------------------------------------------------------------------------
# llama.cpp configuration
# -----------------------------------------------------------------------------

set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/llama.cpp)

set(BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)

set(GGML_OPENMP OFF)
set(GGML_METAL OFF)
set(GGML_CUDA OFF)

# Keep Vulkan/GPU support enabled for llama.cpp when Android Vulkan exists.
find_library(VULKAN_LIB vulkan)

if(VULKAN_LIB)
    set(GGML_VULKAN ON CACHE BOOL "Enable llama.cpp Vulkan backend" FORCE)
    message(STATUS "Android Vulkan library found — llama.cpp GPU acceleration enabled")
    message(STATUS "Android Vulkan lib: ${VULKAN_LIB}")
else()
    set(GGML_VULKAN OFF CACHE BOOL "Enable llama.cpp Vulkan backend" FORCE)
    message(STATUS "Android Vulkan library not found — llama.cpp CPU only")
endif()

add_subdirectory(${LLAMA_DIR} llama_build)

add_library(llamachat SHARED llama_jni.cpp)

target_include_directories(llamachat PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

if(DEFINED HOST_VULKAN_INCLUDE_DIR AND EXISTS "${HOST_VULKAN_INCLUDE_DIR}/vulkan/vulkan.hpp")
    target_include_directories(llamachat PRIVATE "${HOST_VULKAN_INCLUDE_DIR}")
endif()

target_link_libraries(llamachat
    llama
    ggml
    android
    log
)

if(VULKAN_LIB)
    target_link_libraries(llamachat ${VULKAN_LIB})
endif()

# -----------------------------------------------------------------------------
# PocketMind image engine.
#
# This stays separate from libllamachat.so so chat/LLM does not break while
# offline Stable Diffusion is added.
#
# Safe behavior:
# - If app/src/main/cpp/stable-diffusion.cpp exists, this builds real SD support.
# - If it does not exist, this still builds the native preview engine.
# -----------------------------------------------------------------------------

set(SDCPP_DIR ${CMAKE_SOURCE_DIR}/stable-diffusion.cpp)
set(POCKETMIND_HAS_SDCPP OFF)

if(EXISTS "${SDCPP_DIR}/CMakeLists.txt" AND EXISTS "${SDCPP_DIR}/include/stable-diffusion.h")
    message(STATUS "stable-diffusion.cpp found — enabling real offline image generation")

    # Keep this conservative first: CPU-only SD engine.
    # llama.cpp Vulkan remains separate above.
    set(SD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SD_WEBP OFF CACHE BOOL "" FORCE)
    set(SD_WEBM OFF CACHE BOOL "" FORCE)
    set(SD_CUDA OFF CACHE BOOL "" FORCE)
    set(SD_METAL OFF CACHE BOOL "" FORCE)
    set(SD_OPENCL OFF CACHE BOOL "" FORCE)
    set(SD_SYCL OFF CACHE BOOL "" FORCE)
    set(SD_VULKAN OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    add_subdirectory(${SDCPP_DIR} sd_build)
    set(POCKETMIND_HAS_SDCPP ON)
else()
    message(STATUS "stable-diffusion.cpp not found — keeping safe native preview image engine")
endif()

add_library(pocketmindimage SHARED image_jni.cpp)

target_link_libraries(pocketmindimage
    android
    log
)

if(POCKETMIND_HAS_SDCPP)
    target_compile_definitions(pocketmindimage PRIVATE POCKETMIND_HAS_SDCPP=1)
    target_include_directories(pocketmindimage PRIVATE
        ${SDCPP_DIR}/include
        ${SDCPP_DIR}/src
    )
    target_link_libraries(pocketmindimage stable-diffusion)
endif()