# Remove stale decoder test artifacts from the working directory (build dir).
# Invoked via: cmake -P cleanup_artifacts.cmake
# Note: *.j2c is excluded — encoder round-trip tests produce j2c files
# that are consumed by their corresponding decode steps.
foreach(_ext pgx ppm pgm)
  file(GLOB _files "*.${_ext}")
  if(_files)
    file(REMOVE ${_files})
  endif()
endforeach()
