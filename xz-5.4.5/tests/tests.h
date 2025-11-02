///////////////////////////////////////////////////////////////////////////////
//
/// \file       tests.h
/// \brief      Common definitions for test applications
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef LZMA_TESTS_H
#define LZMA_TESTS_H

#include "sysdefs.h"
#include "tuklib_integer.h"
#include "lzma.h"
#include "tuktest.h"


// Invalid value for the lzma_check enumeration. This must be positive
// but small enough to fit into signed char since the underlying type might
// one some platform be a signed char.
//
// Don't put LZMA_ at the beginning of the name so that it is obvious that
// this constant doesn't come from the API headers.
#define INVALID_LZMA_CHECK_ID ((lzma_check)(LZMA_CHECK_ID_MAX + 1))


// This table and macro allow getting more readable error messages when
// comparing the lzma_ret enumeration values.
static const char enum_strings_lzma_ret[][24] = {
	"LZMA_OK",
	"LZMA_STREAM_END",
	"LZMA_NO_CHECK",
	"LZMA_UNSUPPORTED_CHECK",
	"LZMA_GET_CHECK",
	"LZMA_MEM_ERROR",
	"LZMA_MEMLIMIT_ERROR",
	"LZMA_FORMAT_ERROR",
	"LZMA_OPTIONS_ERROR",
	"LZMA_DATA_ERROR",
	"LZMA_BUF_ERROR",
	"LZMA_PROG_ERROR",
	"LZMA_SEEK_NEEDED",
};

#define assert_lzma_ret(test_expr, ref_val) \
	assert_enum_eq(test_expr, ref_val, enum_strings_lzma_ret)


static const char enum_strings_lzma_check[][24] = {
	"LZMA_CHECK_NONE",
	"LZMA_CHECK_CRC32",
	"LZMA_CHECK_UNKNOWN_2",
	"LZMA_CHECK_UNKNOWN_3",
	"LZMA_CHECK_CRC64",
	"LZMA_CHECK_UNKNOWN_5",
	"LZMA_CHECK_UNKNOWN_6",
	"LZMA_CHECK_UNKNOWN_7",
	"LZMA_CHECK_UNKNOWN_8",
	"LZMA_CHECK_UNKNOWN_9",
	"LZMA_CHECK_SHA256",
	"LZMA_CHECK_UNKNOWN_11",
	"LZMA_CHECK_UNKNOWN_12",
	"LZMA_CHECK_UNKNOWN_13",
	"LZMA_CHECK_UNKNOWN_14",
	"LZMA_CHECK_UNKNOWN_15",
};

#define assert_lzma_check(test_expr, ref_val) \
	assert_enum_eq(test_expr, ref_val, enum_strings_lzma_check)

#endif
