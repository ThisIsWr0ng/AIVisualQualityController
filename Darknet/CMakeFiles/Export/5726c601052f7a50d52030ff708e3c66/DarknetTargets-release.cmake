#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Darknet::dark" for configuration "Release"
set_property(TARGET Darknet::dark APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Darknet::dark PROPERTIES
  IMPORTED_IMPLIB_RELEASE "C:/Darknet/darknet.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "CuDNN::CuDNN"
  IMPORTED_LOCATION_RELEASE "C:/Darknet/darknet.dll"
  )

list(APPEND _cmake_import_check_targets Darknet::dark )
list(APPEND _cmake_import_check_files_for_Darknet::dark "C:/Darknet/darknet.lib" "C:/Darknet/darknet.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
