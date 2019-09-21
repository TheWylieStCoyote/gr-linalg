INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_LINALG linalg)

FIND_PATH(
    LINALG_INCLUDE_DIRS
    NAMES linalg/api.h
    HINTS $ENV{LINALG_DIR}/include
        ${PC_LINALG_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    LINALG_LIBRARIES
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

include("${CMAKE_CURRENT_LIST_DIR}/linalgTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LINALG DEFAULT_MSG LINALG_LIBRARIES LINALG_INCLUDE_DIRS)
MARK_AS_ADVANCED(LINALG_LIBRARIES LINALG_INCLUDE_DIRS)
