# Sed script that removes the POT-Creation-Date line in the header entry
# from a POT file.
#
# Copyright (C) 2002 Free Software Foundation, Inc.
# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved.  This file is offered as-is,
# without any warranty.
#
# The distinction between the first and the following occurrences of the
# pattern is achieved by looking at the hold space.
/^"POT-Creation-Date: .*"$/{
x
# Test if the hold space is empty.
s/P/P/
ta
# Yes it was empty. First occurrence. Remove the line.
g
d
bb
:a
# The hold space was nonempty. Following occurrences. Do nothing.
x
:b
}
