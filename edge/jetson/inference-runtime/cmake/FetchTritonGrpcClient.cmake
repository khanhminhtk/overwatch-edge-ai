include(ExternalProject)

set(TRITON_CLIENT_GIT_TAG "r23.06" CACHE STRING "Triton client release tag")
set(TRITON_CLIENT_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/triton-client-install")
set(TRITON_CLIENT_INCLUDE_DIR "${TRITON_CLIENT_INSTALL_DIR}/include")
set(TRITON_CLIENT_LIB_DIR "${TRITON_CLIENT_INSTALL_DIR}/lib")

if(NOT TARGET TritonClient::grpcclient)
  # CMake validates imported include directories during configure, before the
  # external project has installed anything into the prefix.
  file(MAKE_DIRECTORY "${TRITON_CLIENT_INCLUDE_DIR}" "${TRITON_CLIENT_LIB_DIR}")

  ExternalProject_Add(
    triton_client_external
    GIT_REPOSITORY https://github.com/triton-inference-server/client.git
    GIT_TAG ${TRITON_CLIENT_GIT_TAG}
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
    PREFIX "${CMAKE_BINARY_DIR}/_deps/triton-client"
    INSTALL_DIR "${TRITON_CLIENT_INSTALL_DIR}"
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${TRITON_CLIENT_INSTALL_DIR}
      -DTRITON_ENABLE_CC_GRPC=ON
      -DTRITON_ENABLE_CC_HTTP=OFF
      -DTRITON_ENABLE_PYTHON_HTTP=OFF
      -DTRITON_ENABLE_PYTHON_GRPC=OFF
      -DTRITON_ENABLE_JAVA_HTTP=OFF
      -DTRITON_ENABLE_TESTS=OFF
      -DTRITON_ENABLE_EXAMPLES=OFF
      -DTRITON_ENABLE_GPU=OFF
      -DTRITON_ENABLE_ZLIB=ON
    BUILD_BYPRODUCTS
      "${TRITON_CLIENT_LIB_DIR}/libgrpcclient.so"
  )

  add_library(TritonClient::grpcclient SHARED IMPORTED GLOBAL)
  set_target_properties(
    TritonClient::grpcclient
    PROPERTIES
      IMPORTED_LOCATION "${TRITON_CLIENT_LIB_DIR}/libgrpcclient.so"
      INTERFACE_INCLUDE_DIRECTORIES "${TRITON_CLIENT_INCLUDE_DIR}"
  )
endif()
