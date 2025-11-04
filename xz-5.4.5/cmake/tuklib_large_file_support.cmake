#
# tuklib_large_file_support.cmake
#
# If off_t is less than 64 bits by default and -D_FILE_OFFSET_BITS=64
# makes off_t become 64-bit, the CMake option LARGE_FILE_SUPPORT is
# provided (ON by default) and -D_FILE_OFFSET_BITS=64 is added to
# the compile definitions if LARGE_FILE_SUPPORT is ON.
#
# Author: Lasse Collin
#
# This file has been put into the public domain.
# You can do whatever you want with this file.
#

include("${CMAKE_CURRENT_LIST_DIR}/tuklib_common.cmake")
include(CheckCSourceCompiles)

function(tuklib_large_file_support TARGET_OR_ALL)
    # MSVC must be handled specially in the C code.
    if(MSVC)
        return()
    endif()

    set(TUKLIB_LARGE_FILE_SUPPORT_TEST
            "#include <sys/types.h>
            int foo[sizeof(off_t) >= 8 ? 1 : -1];
            int main(void) { return 0; }")

    check_c_source_compiles("${TUKLIB_LARGE_FILE_SUPPORT_TEST}"
                            TUKLIB_LARGE_FILE_SUPPORT_BY_DEFAULT)

    if(NOT TUKLIB_LARGE_FILE_SUPPORT_BY_DEFAULT)
        cmake_push_check_state()
        # This needs -D.
        list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D_FILE_OFFSET_BITS=64")
        check_c_source_compiles("${TUKLIB_LARGE_FILE_SUPPORT_TEST}"
                                TUKLIB_LARGE_FILE_SUPPORT_WITH_FOB64)
        cmake_pop_check_state()
    endif()

    if(TUKLIB_LARGE_FILE_SUPPORT_WITH_FOB64)
        # Show the option only when _FILE_OFFSET_BITS=64 affects sizeof(off_t).
        option(LARGE_FILE_SUPPORT
               "Use -D_FILE_OFFSET_BITS=64 to support files larger than 2 GiB."
               ON)

        if(LARGE_FILE_SUPPORT)
            # This must not use -D.
            tuklib_add_definitions("${TARGET_OR_ALL}" "_FILE_OFFSET_BITS=64")
        endif()
    endif()
endfunction()
