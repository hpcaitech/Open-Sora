///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_filter_str.c
/// \brief      Tests Filter string functions
//
//  Author:    Jia Tan
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"


static void
test_lzma_str_to_filters(void)
{
	lzma_filter filters[LZMA_FILTERS_MAX + 1];
	int error_pos;

	// Test with NULL string.
	assert_true(lzma_str_to_filters(NULL, &error_pos, filters, 0,
			NULL) != NULL);

	// Test with NULL filter array.
	assert_true(lzma_str_to_filters("lzma2", &error_pos, NULL, 0,
			NULL) != NULL);

	// Test with unsupported flags.
	assert_true(lzma_str_to_filters("lzma2", &error_pos, filters,
			UINT32_MAX, NULL) != NULL);

	assert_true(lzma_str_to_filters("lzma2", &error_pos, filters,
			LZMA_STR_NO_SPACES << 1, NULL) != NULL);

	assert_true(lzma_str_to_filters("lzma2", &error_pos, filters,
			LZMA_STR_NO_SPACES, NULL) != NULL);

	// Test with empty string.
	assert_true(lzma_str_to_filters("", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 0);

	// Test with invalid filter name and missing filter name.
	assert_true(lzma_str_to_filters("lzma2 abcd", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 6);

	assert_true(lzma_str_to_filters("lzma2--abcd", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 7);

	assert_true(lzma_str_to_filters("lzma2--", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 7);

	// Test LZMA_STR_ALL_FILTERS flag (should work with LZMA1 if built).
#if defined(HAVE_ENCODER_LZMA1) || defined(HAVE_DECODER_LZMA1)
	// Using LZMA1 as a Filter should fail without LZMA_STR_ALL_FILTERS.
	assert_true(lzma_str_to_filters("lzma1", &error_pos, filters,
			0, NULL) != NULL);
	assert_int_eq(error_pos, 0);

	assert_true(lzma_str_to_filters("lzma1", &error_pos, filters,
			LZMA_STR_ALL_FILTERS, NULL) == NULL);

	// Verify Filters array IDs are correct. The array should contain
	// only two elements:
	// 1. LZMA1 Filter
	// 2. LZMA_VLI_UNKNOWN filter array terminator
	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA1);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);
#endif

	// Test LZMA_STR_NO_VALIDATION flag. This should allow having the
	// same Filter multiple times in the chain and having a non-last
	// Filter like lzma2 appear before another Filter.
	// Without the flag, "lzma2 lzma2" must fail.
	assert_true(lzma_str_to_filters("lzma2 lzma2", &error_pos, filters,
			0, NULL) != NULL);

	assert_true(lzma_str_to_filters("lzma2 lzma2", &error_pos, filters,
			LZMA_STR_NO_VALIDATION, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[2].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	// Should fail with invalid Filter options (lc + lp must be <= 4).
	assert_true(lzma_str_to_filters("lzma2:lc=3,lp=3", &error_pos, filters,
			LZMA_STR_NO_VALIDATION, NULL) != NULL);

	// Test invalid option name.
	assert_true(lzma_str_to_filters("lzma2:foo=1,bar=2", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 6);

	// Test missing option value.
	assert_true(lzma_str_to_filters("lzma2:lc=", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 9);

	assert_true(lzma_str_to_filters("lzma2:=,pb=1", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 6);

	// Test unsupported preset value.
	assert_true(lzma_str_to_filters("-10", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 2);

	assert_true(lzma_str_to_filters("-5f", &error_pos,
			filters, 0, NULL) != NULL);
	assert_int_eq(error_pos, 2);

	// Test filter chain too long.
	assert_true(lzma_str_to_filters("lzma2 lzma2 lzma2 lzma2 lzma2",
			&error_pos, filters, LZMA_STR_NO_VALIDATION,
			NULL) != NULL);
	assert_int_eq(error_pos, 24);

#if defined(HAVE_ENCODER_LZMA1) || defined(HAVE_DECODER_LZMA1)
	// Should fail with a Filter not supported in the .xz format (lzma1).
	assert_true(lzma_str_to_filters("lzma1", &error_pos, filters,
			LZMA_STR_NO_VALIDATION, NULL) != NULL);
#endif

	// Test setting options with the "=" format.
	assert_true(lzma_str_to_filters("lzma2=dict=4096,lc=2,lp=2,pb=1,"
			"mode=fast,nice=3,mf=hc3,depth=10", &error_pos,
			filters, 0, NULL) == NULL);
	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_options_lzma *opts = filters[0].options;
	assert_uint_eq(opts->dict_size, 4096);
	assert_uint_eq(opts->lc, 2);
	assert_uint_eq(opts->lp, 2);
	assert_uint_eq(opts->pb, 1);
	assert_uint_eq(opts->mode, LZMA_MODE_FAST);
	assert_uint_eq(opts->nice_len, 3);
	assert_uint_eq(opts->mf, LZMA_MF_HC3);
	assert_uint_eq(opts->depth, 10);

	lzma_filters_free(filters, NULL);

#if defined(HAVE_ENCODER_X86) || defined(HAVE_DECODER_X86)
	// Test BCJ Filter options.
	assert_true(lzma_str_to_filters("x86:start=16", &error_pos, filters,
			LZMA_STR_NO_VALIDATION, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_X86);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_options_bcj *bcj_opts = filters[0].options;
	assert_uint_eq(bcj_opts->start_offset, 16);

	lzma_filters_free(filters, NULL);
#endif

#if defined(HAVE_ENCODER_DELTA) || defined(HAVE_DECODER_DELTA)
	// Test Delta Filter options.
	assert_true(lzma_str_to_filters("delta:dist=20", &error_pos, filters,
			LZMA_STR_NO_VALIDATION, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_DELTA);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_options_delta *delta_opts = filters[0].options;
	assert_uint_eq(delta_opts->dist, 20);

	lzma_filters_free(filters, NULL);
#endif

	// Test skipping leading spaces.
	assert_true(lzma_str_to_filters("    lzma2", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	// Test skipping trailing spaces.
	assert_true(lzma_str_to_filters("lzma2    ", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	// Test with "--" instead of space separating.
	assert_true(lzma_str_to_filters("lzma2--lzma2", &error_pos, filters,
			LZMA_STR_NO_VALIDATION, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[2].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	// Test preset with and without leading "-", and with "e".
	assert_true(lzma_str_to_filters("-3", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	assert_true(lzma_str_to_filters("4", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	assert_true(lzma_str_to_filters("9e", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	// Test using a preset as an lzma2 option.
	assert_true(lzma_str_to_filters("lzma2:preset=9e", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	lzma_filters_free(filters, NULL);

	// Test setting dictionary size with invalid modifier suffix.
	assert_true(lzma_str_to_filters("lzma2:dict=4096ZiB", &error_pos, filters,
			0, NULL) != NULL);

	assert_true(lzma_str_to_filters("lzma2:dict=4096KiBs", &error_pos, filters,
			0, NULL) != NULL);

	// Test option that cannot have multiplier modifier.
	assert_true(lzma_str_to_filters("lzma2:pb=1k", &error_pos, filters,
			0, NULL) != NULL);

	// Test option value too large.
	assert_true(lzma_str_to_filters("lzma2:dict=4096GiB", &error_pos, filters,
			0, NULL) != NULL);

	// Test valid uses of multiplier modifiers (k,m,g).
	assert_true(lzma_str_to_filters("lzma2:dict=4096KiB", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	opts = filters[0].options;
	assert_uint_eq(opts->dict_size, 4096 << 10);

	lzma_filters_free(filters, NULL);

	assert_true(lzma_str_to_filters("lzma2:dict=40Mi", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	opts = filters[0].options;
	assert_uint_eq(opts->dict_size, 40 << 20);

	lzma_filters_free(filters, NULL);

	assert_true(lzma_str_to_filters("lzma2:dict=1g", &error_pos, filters,
			0, NULL) == NULL);

	assert_uint_eq(filters[0].id, LZMA_FILTER_LZMA2);
	assert_uint_eq(filters[1].id, LZMA_VLI_UNKNOWN);

	opts = filters[0].options;
	assert_uint_eq(opts->dict_size, 1 << 30);

	lzma_filters_free(filters, NULL);
}


static void
test_lzma_str_from_filters(void)
{
	lzma_filter filters[LZMA_FILTERS_MAX];
	filters[0].id = LZMA_VLI_UNKNOWN;

	char *output_str = NULL;

	// Test basic NULL inputs.
	assert_lzma_ret(lzma_str_from_filters(NULL, filters, 0, NULL),
			LZMA_PROG_ERROR);

	assert_lzma_ret(lzma_str_from_filters(&output_str, NULL, 0, NULL),
			LZMA_PROG_ERROR);

	// Test with empty filters array.
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters, 0, NULL),
			LZMA_OPTIONS_ERROR);

	// Create a simple filter array only containing an LZMA2 Filter.
	assert_true(lzma_str_to_filters("lzma2", NULL, filters, 0, NULL)
			== NULL);

	// Test with bad flags.
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_ALL_FILTERS, NULL), LZMA_OPTIONS_ERROR);

	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_NO_VALIDATION, NULL), LZMA_OPTIONS_ERROR);

	// Test with no flags.
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters, 0, NULL),
			LZMA_OK);

	assert_str_eq(output_str, "lzma2");
	free(output_str);

	// Test LZMA_STR_ENCODER flag.
	// Only the the return value is checked since the actual string
	// may change in the future (even though it is unlikely).
	// The order of options or the inclusion of new options could
	// cause a change in output, so we will avoid hardcoding an
	// expected result.
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_ENCODER, NULL), LZMA_OK);
	free(output_str);

	// Test LZMA_STR_DECODER flag.
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_DECODER, NULL), LZMA_OK);
	free(output_str);

	// Test LZMA_STR_GETOPT_LONG flag.
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_GETOPT_LONG, NULL), LZMA_OK);
	free(output_str);

	// Test LZMA_STR_NO_SPACES flag.
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_NO_SPACES, NULL), LZMA_OK);

	// Check to be sure there are no spaces.
	assert_true(strchr(output_str, ' ') == NULL);

	free(output_str);

	lzma_filters_free(filters, NULL);

#if defined(HAVE_ENCODER_X86) || defined(HAVE_DECODER_X86)
	assert_true(lzma_str_to_filters("x86 lzma2", NULL, filters, 0, NULL)
			== NULL);

	assert_lzma_ret(lzma_str_from_filters(&output_str, filters, 0, NULL),
			LZMA_OK);

	assert_str_eq(output_str, "x86 lzma2");

	free(output_str);

	// Test setting BCJ option to NULL.
	assert_false(filters[0].options == NULL);
	free(filters[0].options);

	filters[0].options = NULL;

	assert_lzma_ret(lzma_str_from_filters(&output_str, filters, 0, NULL),
			LZMA_OK);

	assert_str_eq(output_str, "x86 lzma2");

	lzma_filters_free(filters, NULL);
	free(output_str);
#endif

	lzma_options_lzma opts;
	assert_false(lzma_lzma_preset(&opts, LZMA_PRESET_DEFAULT));
	// Test with too many Filters (array terminated after 4+ filters).
	lzma_filter oversized_filters[LZMA_FILTERS_MAX + 2];

	for (uint32_t i = 0; i < ARRAY_SIZE(oversized_filters) - 1; i++) {
		oversized_filters[i].id = LZMA_FILTER_LZMA2;
		oversized_filters[i].options = &opts;
	}

	oversized_filters[LZMA_FILTERS_MAX + 1].id = LZMA_VLI_UNKNOWN;
	oversized_filters[LZMA_FILTERS_MAX + 1].options = NULL;

	assert_lzma_ret(lzma_str_from_filters(&output_str, oversized_filters,
			0, NULL), LZMA_OPTIONS_ERROR);

	// Test with NULL filter options (when they cannot be NULL).
	filters[0].id = LZMA_FILTER_LZMA2;
	filters[0].options = NULL;
	filters[1].id = LZMA_VLI_UNKNOWN;

	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_ENCODER, NULL), LZMA_OPTIONS_ERROR);

	// Test with bad Filter ID.
	filters[0].id = LZMA_VLI_UNKNOWN - 1;
	assert_lzma_ret(lzma_str_from_filters(&output_str, filters,
			LZMA_STR_ENCODER, NULL), LZMA_OPTIONS_ERROR);
}


static const char supported_encoders[][9] = {
	"lzma2",
#ifdef HAVE_ENCODER_X86
	"x86",
#endif
#ifdef HAVE_ENCODER_POWERPC
	"powerpc",
#endif
#ifdef HAVE_ENCODER_IA64
	"ia64",
#endif
#ifdef HAVE_ENCODER_ARM
	"arm",
#endif
#ifdef HAVE_ENCODER_ARMTHUMB
	"armthumb",
#endif
#ifdef HAVE_ENCODER_SPARC
	"sparc",
#endif
#ifdef HAVE_ENCODER_ARM64
	"arm64",
#endif
#ifdef HAVE_ENCODER_DELTA
	"delta",
#endif
};

static const char supported_decoders[][9] = {
	"lzma2",
#ifdef HAVE_DECODER_X86
	"x86",
#endif
#ifdef HAVE_DECODER_POWERPC
	"powerpc",
#endif
#ifdef HAVE_DECODER_IA64
	"ia64",
#endif
#ifdef HAVE_DECODER_ARM
	"arm",
#endif
#ifdef HAVE_DECODER_ARMTHUMB
	"armthumb",
#endif
#ifdef HAVE_DECODER_SPARC
	"sparc",
#endif
#ifdef HAVE_DECODER_ARM64
	"arm64",
#endif
#ifdef HAVE_DECODER_DELTA
	"delta",
#endif
};

static const char supported_filters[][9] = {
	"lzma2",
#if defined(HAVE_ENCODER_X86) || defined(HAVE_DECODER_X86)
	"x86",
#endif
#if defined(HAVE_ENCODER_POWERPC) || defined(HAVE_DECODER_POWERPC)
	"powerpc",
#endif
#if defined(HAVE_ENCODER_IA64) || defined(HAVE_DECODER_IA64)
	"ia64",
#endif
#if defined(HAVE_ENCODER_ARM) || defined(HAVE_DECODER_ARM)
	"arm",
#endif
#if defined(HAVE_ENCODER_ARMTHUMB) || defined(HAVE_DECODER_ARMTHUMB)
	"armthumb",
#endif
#if defined(HAVE_ENCODER_SPARC) || defined(HAVE_DECODER_SPARC)
	"sparc",
#endif
#if defined(HAVE_ENCODER_ARM64) || defined(HAVE_DECODER_ARM64)
	"arm64",
#endif
#if defined(HAVE_ENCODER_DELTA) || defined(HAVE_DECODER_DELTA)
	"delta",
#endif
};


static void
test_lzma_str_list_filters(void)
{
	// Test with basic NULL inputs.
	assert_lzma_ret(lzma_str_list_filters(NULL, LZMA_VLI_UNKNOWN, 0,
			NULL), LZMA_PROG_ERROR);

	char *str = NULL;

	// Test with bad flags.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN,
			LZMA_STR_NO_VALIDATION , NULL), LZMA_OPTIONS_ERROR);

	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN,
			LZMA_STR_NO_SPACES, NULL), LZMA_OPTIONS_ERROR);

	// Test with bad Filter ID.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN - 1,
			0, NULL), LZMA_OPTIONS_ERROR);

	// Test LZMA_STR_ENCODER flag.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN,
			LZMA_STR_ENCODER, NULL), LZMA_OK);

	for (uint32_t i = 0; i < ARRAY_SIZE(supported_encoders); i++)
		assert_str_contains(str, supported_encoders[i]);

	free(str);

	// Test LZMA_STR_DECODER flag.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN,
			LZMA_STR_DECODER, NULL), LZMA_OK);

	for (uint32_t i = 0; i < ARRAY_SIZE(supported_decoders); i++)
		assert_str_contains(str, supported_decoders[i]);

	free(str);

	// Test LZMA_STR_GETOPT_LONG flag.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN,
			LZMA_STR_GETOPT_LONG, NULL), LZMA_OK);

	free(str);

	// Test LZMA_STR_ALL_FILTERS flag.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN,
			LZMA_STR_ALL_FILTERS, NULL), LZMA_OK);
#if defined(HAVE_ENCODER_LZMA1) || defined(HAVE_DECODER_LZMA1)
	// With the flag, the string should contain the LZMA1 Filter.
	assert_str_contains(str, "lzma1");

	free(str);

	// If a non .xz filter is specified, it should still list the Filter.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_FILTER_LZMA1,
			0, NULL), LZMA_OK);
	assert_str_eq(str, "lzma1");
#endif
	free(str);

	// Test with no flags.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_VLI_UNKNOWN,
			0, NULL), LZMA_OK);

	for (uint32_t i = 0; i < ARRAY_SIZE(supported_filters); i++)
		assert_str_contains(str, supported_filters[i]);

	assert_str_doesnt_contain(str, "lzma1");

	free(str);

	// Test providing a Filter ID.
	assert_lzma_ret(lzma_str_list_filters(&str, LZMA_FILTER_LZMA2,
			LZMA_STR_ALL_FILTERS, NULL), LZMA_OK);
	assert_str_eq(str, "lzma2");

	free(str);
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);

	tuktest_run(test_lzma_str_to_filters);
	tuktest_run(test_lzma_str_from_filters);
	tuktest_run(test_lzma_str_list_filters);

	return tuktest_end();
}
