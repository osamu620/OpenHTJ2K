# Fails unless FILE holds at most MAX bytes and, when MIN is given, at least MIN.
# Invoked via: cmake -DFILE=<path> -DMAX=<bytes> [-DMIN=<bytes>] -P check_file_size.cmake
if(NOT EXISTS "${FILE}")
  message(FATAL_ERROR "${FILE} does not exist")
endif()
# file(SIZE) needs CMake 3.14; the project supports 3.13.
file(READ "${FILE}" _hex HEX)
string(LENGTH "${_hex}" _digits)
math(EXPR _size "${_digits} / 2")
if(_size GREATER MAX)
  message(FATAL_ERROR "${FILE} takes ${_size} bytes, more than ${MAX}")
endif()
if(DEFINED MIN AND _size LESS MIN)
  message(FATAL_ERROR "${FILE} takes ${_size} bytes, fewer than ${MIN}")
endif()
message(STATUS "${FILE}: ${_size} bytes")
