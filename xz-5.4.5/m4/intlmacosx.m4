# intlmacosx.m4 serial 8 (gettext-0.20.2)
dnl Copyright (C) 2004-2014, 2016, 2019-2022 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.
dnl
dnl This file can be used in projects which are not available under
dnl the GNU General Public License or the GNU Lesser General Public
dnl License but which still want to provide support for the GNU gettext
dnl functionality.
dnl Please note that the actual code of the GNU gettext library is covered
dnl by the GNU Lesser General Public License, and the rest of the GNU
dnl gettext package is covered by the GNU General Public License.
dnl They are *not* in the public domain.

dnl Checks for special options needed on Mac OS X.
dnl Defines INTL_MACOSX_LIBS.
AC_DEFUN([gt_INTL_MACOSX],
[
  dnl Check for API introduced in Mac OS X 10.4.
  AC_CACHE_CHECK([for CFPreferencesCopyAppValue],
    [gt_cv_func_CFPreferencesCopyAppValue],
    [gt_save_LIBS="$LIBS"
     LIBS="$LIBS -Wl,-framework -Wl,CoreFoundation"
     AC_LINK_IFELSE(
       [AC_LANG_PROGRAM(
          [[#include <CoreFoundation/CFPreferences.h>]],
          [[CFPreferencesCopyAppValue(NULL, NULL)]])],
       [gt_cv_func_CFPreferencesCopyAppValue=yes],
       [gt_cv_func_CFPreferencesCopyAppValue=no])
     LIBS="$gt_save_LIBS"])
  if test $gt_cv_func_CFPreferencesCopyAppValue = yes; then
    AC_DEFINE([HAVE_CFPREFERENCESCOPYAPPVALUE], [1],
      [Define to 1 if you have the Mac OS X function CFPreferencesCopyAppValue in the CoreFoundation framework.])
  fi
  dnl Don't check for the API introduced in Mac OS X 10.5, CFLocaleCopyCurrent,
  dnl because in macOS 10.13.4 it has the following behaviour:
  dnl When two or more languages are specified in the
  dnl "System Preferences > Language & Region > Preferred Languages" panel,
  dnl it returns en_CC where CC is the territory (even when English is not among
  dnl the preferred languages!).  What we want instead is what
  dnl CFLocaleCopyCurrent returned in earlier macOS releases and what
  dnl CFPreferencesCopyAppValue still returns, namely ll_CC where ll is the
  dnl first among the preferred languages and CC is the territory.
  AC_CACHE_CHECK([for CFLocaleCopyPreferredLanguages], [gt_cv_func_CFLocaleCopyPreferredLanguages],
    [gt_save_LIBS="$LIBS"
     LIBS="$LIBS -Wl,-framework -Wl,CoreFoundation"
     AC_LINK_IFELSE(
       [AC_LANG_PROGRAM(
          [[#include <CoreFoundation/CFLocale.h>]],
          [[CFLocaleCopyPreferredLanguages();]])],
       [gt_cv_func_CFLocaleCopyPreferredLanguages=yes],
       [gt_cv_func_CFLocaleCopyPreferredLanguages=no])
     LIBS="$gt_save_LIBS"])
  if test $gt_cv_func_CFLocaleCopyPreferredLanguages = yes; then
    AC_DEFINE([HAVE_CFLOCALECOPYPREFERREDLANGUAGES], [1],
      [Define to 1 if you have the Mac OS X function CFLocaleCopyPreferredLanguages in the CoreFoundation framework.])
  fi
  INTL_MACOSX_LIBS=
  if test $gt_cv_func_CFPreferencesCopyAppValue = yes \
     || test $gt_cv_func_CFLocaleCopyPreferredLanguages = yes; then
    INTL_MACOSX_LIBS="-Wl,-framework -Wl,CoreFoundation"
  fi
  AC_SUBST([INTL_MACOSX_LIBS])
])
