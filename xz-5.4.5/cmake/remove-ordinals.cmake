#############################################################################
#
# remove-ordinals.cmake
#
# Removes the ordinal numbers from a DEF file that has been created by
# GNU ld or LLVM lld option --output-def (when creating a Windows DLL).
# This should be equivalent: sed 's/ \+@ *[0-9]\+//'
#
# Usage:
#
#     cmake -DINPUT_FILE=infile.def.in \
#           -DOUTPUT_FILE=outfile.def \
#           -P remove-ordinals.cmake
#
#############################################################################
#
# Author: Lasse Collin
#
# This file has been put into the public domain.
# You can do whatever you want with this file.
#
#############################################################################

file(READ "${INPUT_FILE}" STR)
string(REGEX REPLACE " +@ *[0-9]+" "" STR "${STR}")
file(WRITE "${OUTPUT_FILE}" "${STR}")
