#!/bin/sh

###############################################################################
#
# Author: Lasse Collin
#
# This file has been put into the public domain.
# You can do whatever you want with this file.
#
###############################################################################

set -e -x

# The following six lines are almost identical to "autoreconf -fi" but faster.
${AUTOPOINT:-autopoint} -f
${LIBTOOLIZE:-libtoolize} -c -f || glibtoolize -c -f
${ACLOCAL:-aclocal} -I m4
${AUTOCONF:-autoconf}
${AUTOHEADER:-autoheader}
${AUTOMAKE:-automake} -acf --foreign

# Generate the translated man pages and the doxygen documentation if the
# "po4a" and "doxygen" tools are available.
# This is *NOT* done by "autoreconf -fi" or when "make" is run.
# Pass --no-po4a or --no-doxygen to this script to skip these steps.
# It can be useful when you know that po4a or doxygen aren't available and
# don't want autogen.sh to exit with non-zero exit status.
generate_po4a="y"
generate_doxygen="y"

for arg in "$@"
do
	case $arg in
		"--no-po4a")
			generate_po4a="n"
			;;

		"--no-doxygen")
			generate_doxygen="n"
			;;
	esac
done

if test "$generate_po4a" != "n"; then
	cd po4a
	sh update-po
	cd ..
fi

if test "$generate_doxygen" != "n"; then
	cd doxygen
	sh update-doxygen
	cd ..
fi

exit 0
