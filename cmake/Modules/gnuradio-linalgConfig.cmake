find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_LINALG gnuradio-linalg)

FIND_PATH(
    GR_LINALG_INCLUDE_DIRS
    NAMES gnuradio/linalg/api.h
    HINTS $ENV{LINALG_DIR}/include
        ${PC_LINALG_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_LINALG_LIBRARIES
    NAMES gnuradio-linalg
    HINTS $ENV{LINALG_DIR}/lib
        ${PC_LINALG_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-linalgTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_LINALG DEFAULT_MSG GR_LINALG_LIBRARIES GR_LINALG_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_LINALG_LIBRARIES GR_LINALG_INCLUDE_DIRS)
