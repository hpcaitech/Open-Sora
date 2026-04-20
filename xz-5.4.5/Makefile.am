##
## Author: Lasse Collin
##
## This file has been put into the public domain.
## You can do whatever you want with this file.
##

# Use -n to prevent gzip from adding a timestamp to the .gz headers.
GZIP_ENV = -9n

DIST_SUBDIRS = lib src po tests debug
SUBDIRS =

if COND_GNULIB
SUBDIRS += lib
endif

SUBDIRS += src po tests

if COND_DOC
dist_doc_DATA = \
	AUTHORS \
	COPYING \
	COPYING.GPLv2 \
	NEWS \
	README \
	THANKS \
	TODO \
	doc/faq.txt \
	doc/history.txt \
	doc/xz-file-format.txt \
	doc/lzma-file-format.txt

examplesdir = $(docdir)/examples
dist_examples_DATA = \
	doc/examples/00_README.txt \
	doc/examples/01_compress_easy.c \
	doc/examples/02_decompress.c \
	doc/examples/03_compress_custom.c \
	doc/examples/04_compress_easy_mt.c \
	doc/examples/Makefile

examplesolddir = $(docdir)/examples_old
dist_examplesold_DATA = \
	doc/examples_old/xz_pipe_comp.c \
	doc/examples_old/xz_pipe_decomp.c

# Install the Doxygen generated documentation if they were built.
install-data-local:
	if test -d "$(srcdir)/doc/api" ; then \
		$(MKDIR_P) "$(DESTDIR)$(docdir)/api" && \
		$(INSTALL_DATA) "$(srcdir)"/doc/api/* \
			"$(DESTDIR)$(docdir)/api"; \
	fi

# Remove the Doxygen generated documentation when uninstalling.
uninstall-local:
	rm -rf "$(DESTDIR)$(docdir)/api"
endif

EXTRA_DIST = \
	po4a \
	extra \
	dos \
	doxygen \
	windows \
	macosx \
	cmake \
	CMakeLists.txt \
	autogen.sh \
	COPYING.GPLv2 \
	COPYING.GPLv3 \
	COPYING.LGPLv2.1 \
	INSTALL.generic \
	PACKAGERS \
	build-aux/manconv.sh \
	build-aux/version.sh

ACLOCAL_AMFLAGS = -I m4

# List of man pages to convert to PDF and plain text in the dist-hook target.
manfiles = \
	src/xz/xz.1 \
	src/xzdec/xzdec.1 \
	src/lzmainfo/lzmainfo.1 \
	src/scripts/xzdiff.1 \
	src/scripts/xzgrep.1 \
	src/scripts/xzless.1 \
	src/scripts/xzmore.1

# Create ChangeLog from output of "git log --date=iso --stat".
# Convert the man pages to PDF and plain text (ASCII only) formats.
dist-hook:
	if test -d "$(srcdir)/.git" && type git > /dev/null 2>&1; then \
		( cd "$(srcdir)" && git log --date=iso --stat \
			6060f7dc76fd6c2a8a1f8e85d0e4d86bb78273e6^..HEAD ) \
			> "$(distdir)/ChangeLog"; \
	fi
	if type groff > /dev/null 2>&1 && type ps2pdf > /dev/null 2>&1; then \
		dest="$(distdir)/doc/man" && \
		$(MKDIR_P) "$$dest/pdf-a4" "$$dest/pdf-letter" "$$dest/txt" && \
		for FILE in $(manfiles); do \
			BASE=`basename $$FILE .1` && \
			sh "$(srcdir)/build-aux/manconv.sh" pdf a4 \
				< "$(srcdir)/$$FILE" \
				> "$$dest/pdf-a4/$$BASE-a4.pdf" && \
			sh "$(srcdir)/build-aux/manconv.sh" pdf letter \
				< "$(srcdir)/$$FILE" \
				> "$$dest/pdf-letter/$$BASE-letter.pdf" && \
			sh "$(srcdir)/build-aux/manconv.sh" ascii \
				< "$(srcdir)/$$FILE" \
				> "$$dest/txt/$$BASE.txt"; \
		done; \
	fi
	if test -d "$(srcdir)/doc/api" ; then \
		$(MKDIR_P) "$(distdir)/doc/api" && \
		$(INSTALL_DATA) "$(srcdir)"/doc/api/* \
			"$(distdir)/doc/api"; \
	fi

# This works with GNU tar and gives cleaner package than normal 'make dist'.
# This also ensures that the man page translations are up to date (dist-hook
# would be too late for that).
mydist:
	sh "$(srcdir)/src/liblzma/validate_map.sh"
	cd "$(srcdir)/po4a" && sh update-po
	cd "$(srcdir)/doxygen" && sh update-doxygen
	VERSION=$(VERSION); \
	if test -d "$(srcdir)/.git" && type git > /dev/null 2>&1; then \
		SNAPSHOT=`cd "$(srcdir)" && git describe --abbrev=4 | cut -b2-`; \
		test -n "$$SNAPSHOT" && VERSION=$$SNAPSHOT; \
	fi; \
	TAR_OPTIONS='--owner=0 --group=0 --numeric-owner --mode=u+rw,go+r-w' \
		$(MAKE) VERSION="$$VERSION" dist-gzip
