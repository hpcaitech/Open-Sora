///////////////////////////////////////////////////////////////////////////////
//
/// \file       string_conversion.c
/// \brief      Conversion of strings to filter chain and vice versa
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "filter_common.h"


/////////////////////
// String building //
/////////////////////

/// How much memory to allocate for strings. For now, no realloc is used
/// so this needs to be big enough even though there of course is
/// an overflow check still.
///
/// FIXME? Using a fixed size is wasteful if the application doesn't free
/// the string fairly quickly but this can be improved later if needed.
#define STR_ALLOC_SIZE 800


typedef struct {
	char *buf;
	size_t pos;
} lzma_str;


static lzma_ret
str_init(lzma_str *str, const lzma_allocator *allocator)
{
	str->buf = lzma_alloc(STR_ALLOC_SIZE, allocator);
	if (str->buf == NULL)
		return LZMA_MEM_ERROR;

	str->pos = 0;
	return LZMA_OK;
}


static void
str_free(lzma_str *str, const lzma_allocator *allocator)
{
	lzma_free(str->buf, allocator);
	return;
}


static bool
str_is_full(const lzma_str *str)
{
	return str->pos == STR_ALLOC_SIZE - 1;
}


static lzma_ret
str_finish(char **dest, lzma_str *str, const lzma_allocator *allocator)
{
	if (str_is_full(str)) {
		// The preallocated buffer was too small.
		// This shouldn't happen as STR_ALLOC_SIZE should
		// be adjusted if new filters are added.
		lzma_free(str->buf, allocator);
		*dest = NULL;
		assert(0);
		return LZMA_PROG_ERROR;
	}

	str->buf[str->pos] = '\0';
	*dest = str->buf;
	return LZMA_OK;
}


static void
str_append_str(lzma_str *str, const char *s)
{
	const size_t len = strlen(s);
	const size_t limit = STR_ALLOC_SIZE - 1 - str->pos;
	const size_t copy_size = my_min(len, limit);

	memcpy(str->buf + str->pos, s, copy_size);
	str->pos += copy_size;
	return;
}


static void
str_append_u32(lzma_str *str, uint32_t v, bool use_byte_suffix)
{
	if (v == 0) {
		str_append_str(str, "0");
	} else {
		// NOTE: Don't use plain "B" because xz and the parser in this
		// file don't support it and at glance it may look like 8
		// (there cannot be a space before the suffix).
		static const char suffixes[4][4] = { "", "KiB", "MiB", "GiB" };

		size_t suf = 0;
		if (use_byte_suffix) {
			while ((v & 1023) == 0
					&& suf < ARRAY_SIZE(suffixes) - 1) {
				v >>= 10;
				++suf;
			}
		}

		// UINT32_MAX in base 10 would need 10 + 1 bytes. Remember
		// that initializing to "" initializes all elements to
		// zero so '\0'-termination gets handled by this.
		char buf[16] = "";
		size_t pos = sizeof(buf) - 1;

		do {
			buf[--pos] = '0' + (v % 10);
			v /= 10;
		} while (v != 0);

		str_append_str(str, buf + pos);
		str_append_str(str, suffixes[suf]);
	}

	return;
}


//////////////////////////////////////////////
// Parsing and stringification declarations //
//////////////////////////////////////////////

/// Maximum length for filter and option names.
/// 11 chars + terminating '\0' + sizeof(uint32_t) = 16 bytes
#define NAME_LEN_MAX 11


/// For option_map.flags: Use .u.map to do convert the input value
/// to an integer. Without this flag, .u.range.{min,max} are used
/// as the allowed range for the integer.
#define OPTMAP_USE_NAME_VALUE_MAP 0x01

/// For option_map.flags: Allow KiB/MiB/GiB in input string and use them in
/// the stringified output if the value is an exact multiple of these.
/// This is used e.g. for LZMA1/2 dictionary size.
#define OPTMAP_USE_BYTE_SUFFIX 0x02

/// For option_map.flags: If the integer value is zero then this option
/// won't be included in the stringified output. It's used e.g. for
/// BCJ filter start offset which usually is zero.
#define OPTMAP_NO_STRFY_ZERO 0x04

/// Possible values for option_map.type. Since OPTMAP_TYPE_UINT32 is 0,
/// it doesn't need to be specified in the initializers as it is
/// the implicit value.
enum {
	OPTMAP_TYPE_UINT32,
	OPTMAP_TYPE_LZMA_MODE,
	OPTMAP_TYPE_LZMA_MATCH_FINDER,
	OPTMAP_TYPE_LZMA_PRESET,
};


/// This is for mapping string values in options to integers.
/// The last element of an array must have "" as the name.
/// It's used e.g. for match finder names in LZMA1/2.
typedef struct {
	const char name[NAME_LEN_MAX + 1];
	const uint32_t value;
} name_value_map;


/// Each filter that has options needs an array of option_map structures.
/// The array doesn't need to be terminated as the functions take the
/// length of the array as an argument.
///
/// When converting a string to filter options structure, option values
/// will be handled in a few different ways:
///
/// (1) If .type equals OPTMAP_TYPE_LZMA_PRESET then LZMA1/2 preset string
///     is handled specially.
///
/// (2) If .flags has OPTMAP_USE_NAME_VALUE_MAP set then the string is
///     converted to an integer using the name_value_map pointed by .u.map.
///     The last element in .u.map must have .name = "" as the terminator.
///
/// (3) Otherwise the string is treated as a non-negative unsigned decimal
///     integer which must be in the range set in .u.range. If .flags has
///     OPTMAP_USE_BYTE_SUFFIX then KiB, MiB, and GiB suffixes are allowed.
///
/// The integer value from (2) or (3) is then stored to filter_options
/// at the offset specified in .offset using the type specified in .type
/// (default is uint32_t).
///
/// Stringifying a filter is done by processing a given number of options
/// in order from the beginning of an option_map array. The integer is
/// read from filter_options at .offset using the type from .type.
///
/// If the integer is zero and .flags has OPTMAP_NO_STRFY_ZERO then the
/// option is skipped.
///
/// If .flags has OPTMAP_USE_NAME_VALUE_MAP set then .u.map will be used
/// to convert the option to a string. If the map doesn't contain a string
/// for the integer value then "UNKNOWN" is used.
///
/// If .flags doesn't have OPTMAP_USE_NAME_VALUE_MAP set then the integer is
/// converted to a decimal value. If OPTMAP_USE_BYTE_SUFFIX is used then KiB,
/// MiB, or GiB suffix is used if the value is an exact multiple of these.
/// Plain "B" suffix is never used.
typedef struct {
	char name[NAME_LEN_MAX + 1];
	uint8_t type;
	uint8_t flags;
	uint16_t offset;

	union {
		struct {
			uint32_t min;
			uint32_t max;
		} range;

		const name_value_map *map;
	} u;
} option_map;


static const char *parse_options(const char **const str, const char *str_end,
		void *filter_options,
		const option_map *const optmap, const size_t optmap_size);


/////////
// BCJ //
/////////

#if defined(HAVE_ENCODER_X86) \
		|| defined(HAVE_DECODER_X86) \
		|| defined(HAVE_ENCODER_ARM) \
		|| defined(HAVE_DECODER_ARM) \
		|| defined(HAVE_ENCODER_ARMTHUMB) \
		|| defined(HAVE_DECODER_ARMTHUMB) \
		|| defined(HAVE_ENCODER_ARM64) \
		|| defined(HAVE_DECODER_ARM64) \
		|| defined(HAVE_ENCODER_POWERPC) \
		|| defined(HAVE_DECODER_POWERPC) \
		|| defined(HAVE_ENCODER_IA64) \
		|| defined(HAVE_DECODER_IA64) \
		|| defined(HAVE_ENCODER_SPARC) \
		|| defined(HAVE_DECODER_SPARC)
static const option_map bcj_optmap[] = {
	{
		.name = "start",
		.flags = OPTMAP_NO_STRFY_ZERO | OPTMAP_USE_BYTE_SUFFIX,
		.offset = offsetof(lzma_options_bcj, start_offset),
		.u.range.min = 0,
		.u.range.max = UINT32_MAX,
	}
};


static const char *
parse_bcj(const char **const str, const char *str_end, void *filter_options)
{
	// filter_options was zeroed on allocation and that is enough
	// for the default value.
	return parse_options(str, str_end, filter_options,
			bcj_optmap, ARRAY_SIZE(bcj_optmap));
}
#endif


///////////
// Delta //
///////////

#if defined(HAVE_ENCODER_DELTA) || defined(HAVE_DECODER_DELTA)
static const option_map delta_optmap[] = {
	{
		.name = "dist",
		.offset = offsetof(lzma_options_delta, dist),
		.u.range.min = LZMA_DELTA_DIST_MIN,
		.u.range.max = LZMA_DELTA_DIST_MAX,
	}
};


static const char *
parse_delta(const char **const str, const char *str_end, void *filter_options)
{
	lzma_options_delta *opts = filter_options;
	opts->type = LZMA_DELTA_TYPE_BYTE;
	opts->dist = LZMA_DELTA_DIST_MIN;

	return parse_options(str, str_end, filter_options,
			delta_optmap, ARRAY_SIZE(delta_optmap));
}
#endif


///////////////////
// LZMA1 & LZMA2 //
///////////////////

/// Help string for presets
#define LZMA12_PRESET_STR "0-9[e]"


static const char *
parse_lzma12_preset(const char **const str, const char *str_end,
		uint32_t *preset)
{
	assert(*str < str_end);
	*preset = (uint32_t)(**str - '0');

	// NOTE: Remember to update LZMA12_PRESET_STR if this is modified!
	while (++*str < str_end) {
		switch (**str) {
		case 'e':
			*preset |= LZMA_PRESET_EXTREME;
			break;

		default:
			return "Unsupported preset flag";
		}
	}

	return NULL;
}


static const char *
set_lzma12_preset(const char **const str, const char *str_end,
		void *filter_options)
{
	uint32_t preset;
	const char *errmsg = parse_lzma12_preset(str, str_end, &preset);
	if (errmsg != NULL)
		return errmsg;

	lzma_options_lzma *opts = filter_options;
	if (lzma_lzma_preset(opts, preset))
		return "Unsupported preset";

	return NULL;
}


static const name_value_map lzma12_mode_map[] = {
	{ "fast",   LZMA_MODE_FAST },
	{ "normal", LZMA_MODE_NORMAL },
	{ "",       0 }
};


static const name_value_map lzma12_mf_map[] = {
	{ "hc3", LZMA_MF_HC3 },
	{ "hc4", LZMA_MF_HC4 },
	{ "bt2", LZMA_MF_BT2 },
	{ "bt3", LZMA_MF_BT3 },
	{ "bt4", LZMA_MF_BT4 },
	{ "",    0 }
};


static const option_map lzma12_optmap[] = {
	{
		.name = "preset",
		.type = OPTMAP_TYPE_LZMA_PRESET,
	}, {
		.name = "dict",
		.flags = OPTMAP_USE_BYTE_SUFFIX,
		.offset = offsetof(lzma_options_lzma, dict_size),
		.u.range.min = LZMA_DICT_SIZE_MIN,
		// FIXME? The max is really max for encoding but decoding
		// would allow 4 GiB - 1 B.
		.u.range.max = (UINT32_C(1) << 30) + (UINT32_C(1) << 29),
	}, {
		.name = "lc",
		.offset = offsetof(lzma_options_lzma, lc),
		.u.range.min = LZMA_LCLP_MIN,
		.u.range.max = LZMA_LCLP_MAX,
	}, {
		.name = "lp",
		.offset = offsetof(lzma_options_lzma, lp),
		.u.range.min = LZMA_LCLP_MIN,
		.u.range.max = LZMA_LCLP_MAX,
	}, {
		.name = "pb",
		.offset = offsetof(lzma_options_lzma, pb),
		.u.range.min = LZMA_PB_MIN,
		.u.range.max = LZMA_PB_MAX,
	}, {
		.name = "mode",
		.type = OPTMAP_TYPE_LZMA_MODE,
		.flags = OPTMAP_USE_NAME_VALUE_MAP,
		.offset = offsetof(lzma_options_lzma, mode),
		.u.map = lzma12_mode_map,
	}, {
		.name = "nice",
		.offset = offsetof(lzma_options_lzma, nice_len),
		.u.range.min = 2,
		.u.range.max = 273,
	}, {
		.name = "mf",
		.type = OPTMAP_TYPE_LZMA_MATCH_FINDER,
		.flags = OPTMAP_USE_NAME_VALUE_MAP,
		.offset = offsetof(lzma_options_lzma, mf),
		.u.map = lzma12_mf_map,
	}, {
		.name = "depth",
		.offset = offsetof(lzma_options_lzma, depth),
		.u.range.min = 0,
		.u.range.max = UINT32_MAX,
	}
};


static const char *
parse_lzma12(const char **const str, const char *str_end, void *filter_options)
{
	lzma_options_lzma *opts = filter_options;

	// It cannot fail.
	const bool preset_ret = lzma_lzma_preset(opts, LZMA_PRESET_DEFAULT);
	assert(!preset_ret);
	(void)preset_ret;

	const char *errmsg = parse_options(str, str_end, filter_options,
			lzma12_optmap, ARRAY_SIZE(lzma12_optmap));
	if (errmsg != NULL)
		return errmsg;

	if (opts->lc + opts->lp > LZMA_LCLP_MAX)
		return "The sum of lc and lp must not exceed 4";

	return NULL;
}


/////////////////////////////////////////
// Generic parsing and stringification //
/////////////////////////////////////////

static const struct {
	/// Name of the filter
	char name[NAME_LEN_MAX + 1];

	/// For lzma_str_to_filters:
	/// Size of the filter-specific options structure.
	uint32_t opts_size;

	/// Filter ID
	lzma_vli id;

	/// For lzma_str_to_filters:
	/// Function to parse the filter-specific options. The filter_options
	/// will already have been allocated using lzma_alloc_zero().
	const char *(*parse)(const char **str, const char *str_end,
			void *filter_options);

	/// For lzma_str_from_filters:
	/// If the flag LZMA_STR_ENCODER is used then the first
	/// strfy_encoder elements of optmap are stringified.
	/// With LZMA_STR_DECODER strfy_decoder is used.
	/// Currently encoders use all options that decoders do but if
	/// that changes then this needs to be changed too, for example,
	/// add a new OPTMAP flag to skip printing some decoder-only options.
	const option_map *optmap;
	uint8_t strfy_encoder;
	uint8_t strfy_decoder;

	/// For lzma_str_from_filters:
	/// If true, lzma_filter.options is allowed to be NULL. In that case,
	/// only the filter name is printed without any options.
	bool allow_null;

} filter_name_map[] = {
#if defined (HAVE_ENCODER_LZMA1) || defined(HAVE_DECODER_LZMA1)
	{ "lzma1",        sizeof(lzma_options_lzma),  LZMA_FILTER_LZMA1,
	  &parse_lzma12,  lzma12_optmap, 9, 5, false },
#endif

#if defined(HAVE_ENCODER_LZMA2) || defined(HAVE_DECODER_LZMA2)
	{ "lzma2",        sizeof(lzma_options_lzma),  LZMA_FILTER_LZMA2,
	  &parse_lzma12,  lzma12_optmap, 9, 2, false },
#endif

#if defined(HAVE_ENCODER_X86) || defined(HAVE_DECODER_X86)
	{ "x86",          sizeof(lzma_options_bcj),   LZMA_FILTER_X86,
	  &parse_bcj,     bcj_optmap, 1, 1, true },
#endif

#if defined(HAVE_ENCODER_ARM) || defined(HAVE_DECODER_ARM)
	{ "arm",          sizeof(lzma_options_bcj),   LZMA_FILTER_ARM,
	  &parse_bcj,     bcj_optmap, 1, 1, true },
#endif

#if defined(HAVE_ENCODER_ARMTHUMB) || defined(HAVE_DECODER_ARMTHUMB)
	{ "armthumb",     sizeof(lzma_options_bcj),   LZMA_FILTER_ARMTHUMB,
	  &parse_bcj,     bcj_optmap, 1, 1, true },
#endif

#if defined(HAVE_ENCODER_ARM64) || defined(HAVE_DECODER_ARM64)
	{ "arm64",        sizeof(lzma_options_bcj),   LZMA_FILTER_ARM64,
	  &parse_bcj,     bcj_optmap, 1, 1, true },
#endif

#if defined(HAVE_ENCODER_POWERPC) || defined(HAVE_DECODER_POWERPC)
	{ "powerpc",      sizeof(lzma_options_bcj),   LZMA_FILTER_POWERPC,
	  &parse_bcj,     bcj_optmap, 1, 1, true },
#endif

#if defined(HAVE_ENCODER_IA64) || defined(HAVE_DECODER_IA64)
	{ "ia64",         sizeof(lzma_options_bcj),   LZMA_FILTER_IA64,
	  &parse_bcj,     bcj_optmap, 1, 1, true },
#endif

#if defined(HAVE_ENCODER_SPARC) || defined(HAVE_DECODER_SPARC)
	{ "sparc",        sizeof(lzma_options_bcj),   LZMA_FILTER_SPARC,
	  &parse_bcj,     bcj_optmap, 1, 1, true },
#endif

#if defined(HAVE_ENCODER_DELTA) || defined(HAVE_DECODER_DELTA)
	{ "delta",        sizeof(lzma_options_delta), LZMA_FILTER_DELTA,
	  &parse_delta,   delta_optmap, 1, 1, false },
#endif
};


/// Decodes options from a string for one filter (name1=value1,name2=value2).
/// Caller must have allocated memory for filter_options already and set
/// the initial default values. This is called from the filter-specific
/// parse_* functions.
///
/// The input string starts at *str and the address in str_end is the first
/// char that is not part of the string anymore. So no '\0' terminator is
/// used. *str is advanced every time something has been decoded successfully.
static const char *
parse_options(const char **const str, const char *str_end,
		void *filter_options,
		const option_map *const optmap, const size_t optmap_size)
{
	while (*str < str_end && **str != '\0') {
		// Each option is of the form name=value.
		// Commas (',') separate options. Extra commas are ignored.
		// Ignoring extra commas makes it simpler if an optional
		// option stored in a shell variable which can be empty.
		if (**str == ',') {
			++*str;
			continue;
		}

		// Find where the next name=value ends.
		const size_t str_len = (size_t)(str_end - *str);
		const char *name_eq_value_end = memchr(*str, ',', str_len);
		if (name_eq_value_end == NULL)
			name_eq_value_end = str_end;

		const char *equals_sign = memchr(*str, '=',
				(size_t)(name_eq_value_end - *str));

		// Fail if the '=' wasn't found or the option name is missing
		// (the first char is '=').
		if (equals_sign == NULL || **str == '=')
			return "Options must be 'name=value' pairs separated "
					"with commas";

		// Reject a too long option name so that the memcmp()
		// in the loop below won't read past the end of the
		// string in optmap[i].name.
		const size_t name_len = (size_t)(equals_sign - *str);
		if (name_len > NAME_LEN_MAX)
			return "Unknown option name";

		// Find the option name from optmap[].
		size_t i = 0;
		while (true) {
			if (i == optmap_size)
				return "Unknown option name";

			if (memcmp(*str, optmap[i].name, name_len) == 0
					&& optmap[i].name[name_len] == '\0')
				break;

			++i;
		}

		// The input string is good at least until the start of
		// the option value.
		*str = equals_sign + 1;

		// The code assumes that the option value isn't an empty
		// string so check it here.
		const size_t value_len = (size_t)(name_eq_value_end - *str);
		if (value_len == 0)
			return "Option value cannot be empty";

		// LZMA1/2 preset has its own parsing function.
		if (optmap[i].type == OPTMAP_TYPE_LZMA_PRESET) {
			const char *errmsg = set_lzma12_preset(str,
					name_eq_value_end, filter_options);
			if (errmsg != NULL)
				return errmsg;

			continue;
		}

		// It's an integer value.
		uint32_t v;
		if (optmap[i].flags & OPTMAP_USE_NAME_VALUE_MAP) {
			// The integer is picked from a string-to-integer map.
			//
			// Reject a too long value string so that the memcmp()
			// in the loop below won't read past the end of the
			// string in optmap[i].u.map[j].name.
			if (value_len > NAME_LEN_MAX)
				return "Invalid option value";

			const name_value_map *map = optmap[i].u.map;
			size_t j = 0;
			while (true) {
				// The array is terminated with an empty name.
				if (map[j].name[0] == '\0')
					return "Invalid option value";

				if (memcmp(*str, map[j].name, value_len) == 0
						&& map[j].name[value_len]
							== '\0') {
					v = map[j].value;
					break;
				}

				++j;
			}
		} else if (**str < '0' || **str > '9') {
			// Note that "max" isn't supported while it is
			// supported in xz. It's not useful here.
			return "Value is not a non-negative decimal integer";
		} else {
			// strtoul() has locale-specific behavior so it cannot
			// be relied on to get reproducible results since we
			// cannot change the locate in a thread-safe library.
			// It also needs '\0'-termination.
			//
			// Use a temporary pointer so that *str will point
			// to the beginning of the value string in case
			// an error occurs.
			const char *p = *str;
			v = 0;
			do {
				if (v > UINT32_MAX / 10)
					return "Value out of range";

				v *= 10;

				const uint32_t add = (uint32_t)(*p - '0');
				if (UINT32_MAX - add < v)
					return "Value out of range";

				v += add;
				++p;
			} while (p < name_eq_value_end
					&& *p >= '0' && *p <= '9');

			if (p < name_eq_value_end) {
				// Remember this position so that it can be
				// used for error messages that are
				// specifically about the suffix. (Out of
				// range values are about the whole value
				// and those error messages point to the
				// beginning of the number part,
				// not to the suffix.)
				const char *multiplier_start = p;

				// If multiplier suffix shouldn't be used
				// then don't allow them even if the value
				// would stay within limits. This is a somewhat
				// unnecessary check but it rejects silly
				// things like lzma2:pb=0MiB which xz allows.
				if ((optmap[i].flags & OPTMAP_USE_BYTE_SUFFIX)
						== 0) {
					*str = multiplier_start;
					return "This option does not support "
						"any integer suffixes";
				}

				uint32_t shift;

				switch (*p) {
				case 'k':
				case 'K':
					shift = 10;
					break;

				case 'm':
				case 'M':
					shift = 20;
					break;

				case 'g':
				case 'G':
					shift = 30;
					break;

				default:
					*str = multiplier_start;
					return "Invalid multiplier suffix "
							"(KiB, MiB, or GiB)";
				}

				++p;

				// Allow "M", "Mi", "MB", "MiB" and the same
				// for the other five characters from the
				// switch-statement above. All are handled
				// as base-2 (perhaps a mistake, perhaps not).
				// Note that 'i' and 'B' are case sensitive.
				if (p < name_eq_value_end && *p == 'i')
					++p;

				if (p < name_eq_value_end && *p == 'B')
					++p;

				// Now we must have no chars remaining.
				if (p < name_eq_value_end) {
					*str = multiplier_start;
					return "Invalid multiplier suffix "
							"(KiB, MiB, or GiB)";
				}

				if (v > (UINT32_MAX >> shift))
					return "Value out of range";

				v <<= shift;
			}

			if (v < optmap[i].u.range.min
					|| v > optmap[i].u.range.max)
				return "Value out of range";
		}

		// Set the value in filter_options. Enums are handled
		// specially since the underlying type isn't the same
		// as uint32_t on all systems.
		void *ptr = (char *)filter_options + optmap[i].offset;
		switch (optmap[i].type) {
		case OPTMAP_TYPE_LZMA_MODE:
			*(lzma_mode *)ptr = (lzma_mode)v;
			break;

		case OPTMAP_TYPE_LZMA_MATCH_FINDER:
			*(lzma_match_finder *)ptr = (lzma_match_finder)v;
			break;

		default:
			*(uint32_t *)ptr = v;
			break;
		}

		// This option has been successfully handled.
		*str = name_eq_value_end;
	}

	// No errors.
	return NULL;
}


/// Finds the name of the filter at the beginning of the string and
/// calls filter_name_map[i].parse() to decode the filter-specific options.
/// The caller must have set str_end so that exactly one filter and its
/// options are present without any trailing characters.
static const char *
parse_filter(const char **const str, const char *str_end, lzma_filter *filter,
		const lzma_allocator *allocator, bool only_xz)
{
	// Search for a colon or equals sign that would separate the filter
	// name from filter options. If neither is found, then the input
	// string only contains a filter name and there are no options.
	//
	// First assume that a colon or equals sign won't be found:
	const char *name_end = str_end;
	const char *opts_start = str_end;

	for (const char *p = *str; p < str_end; ++p) {
		if (*p == ':' || *p == '=') {
			name_end = p;

			// Filter options (name1=value1,name2=value2,...)
			// begin after the colon or equals sign.
			opts_start = p + 1;
			break;
		}
	}

	// Reject a too long filter name so that the memcmp()
	// in the loop below won't read past the end of the
	// string in filter_name_map[i].name.
	const size_t name_len = (size_t)(name_end - *str);
	if (name_len > NAME_LEN_MAX)
		return "Unknown filter name";

	for (size_t i = 0; i < ARRAY_SIZE(filter_name_map); ++i) {
		if (memcmp(*str, filter_name_map[i].name, name_len) == 0
				&& filter_name_map[i].name[name_len] == '\0') {
			if (only_xz && filter_name_map[i].id
					>= LZMA_FILTER_RESERVED_START)
				return "This filter cannot be used in "
						"the .xz format";

			// Allocate the filter-specific options and
			// initialize the memory with zeros.
			void *options = lzma_alloc_zero(
					filter_name_map[i].opts_size,
					allocator);
			if (options == NULL)
				return "Memory allocation failed";

			// Filter name was found so the input string is good
			// at least this far.
			*str = opts_start;

			const char *errmsg = filter_name_map[i].parse(
					str, str_end, options);
			if (errmsg != NULL) {
				lzma_free(options, allocator);
				return errmsg;
			}

			// *filter is modified only when parsing is successful.
			filter->id = filter_name_map[i].id;
			filter->options = options;
			return NULL;
		}
	}

	return "Unknown filter name";
}


/// Converts the string to a filter chain (array of lzma_filter structures).
///
/// *str is advanced every time something has been decoded successfully.
/// This way the caller knows where in the string a possible error occurred.
static const char *
str_to_filters(const char **const str, lzma_filter *filters, uint32_t flags,
		const lzma_allocator *allocator)
{
	const char *errmsg;

	// Skip leading spaces.
	while (**str == ' ')
		++*str;

	if (**str == '\0')
		return "Empty string is not allowed, "
				"try \"6\" if a default value is needed";

	// Detect the type of the string.
	//
	// A string beginning with a digit or a string beginning with
	// one dash and a digit are treated as presets. Trailing spaces
	// will be ignored too (leading spaces were already ignored above).
	//
	// For example, "6", "7  ", "-9e", or "  -3  " are treated as presets.
	// Strings like "-" or "- " aren't preset.
#define MY_IS_DIGIT(c) ((c) >= '0' && (c) <= '9')
	if (MY_IS_DIGIT(**str) || (**str == '-' && MY_IS_DIGIT((*str)[1]))) {
		if (**str == '-')
			++*str;

		// Ignore trailing spaces.
		const size_t str_len = strlen(*str);
		const char *str_end = memchr(*str, ' ', str_len);
		if (str_end != NULL) {
			// There is at least one trailing space. Check that
			// there are no chars other than spaces.
			for (size_t i = 1; str_end[i] != '\0'; ++i)
				if (str_end[i] != ' ')
					return "Unsupported preset";
		} else {
			// There are no trailing spaces. Use the whole string.
			str_end = *str + str_len;
		}

		uint32_t preset;
		errmsg = parse_lzma12_preset(str, str_end, &preset);
		if (errmsg != NULL)
			return errmsg;

		lzma_options_lzma *opts = lzma_alloc(sizeof(*opts), allocator);
		if (opts == NULL)
			return "Memory allocation failed";

		if (lzma_lzma_preset(opts, preset)) {
			lzma_free(opts, allocator);
			return "Unsupported preset";
		}

		filters[0].id = LZMA_FILTER_LZMA2;
		filters[0].options = opts;
		filters[1].id = LZMA_VLI_UNKNOWN;
		filters[1].options = NULL;

		return NULL;
	}

	// Not a preset so it must be a filter chain.
	//
	// If LZMA_STR_ALL_FILTERS isn't used we allow only filters that
	// can be used in .xz.
	const bool only_xz = (flags & LZMA_STR_ALL_FILTERS) == 0;

	// Use a temporary array so that we don't modify the caller-supplied
	// one until we know that no errors occurred.
	lzma_filter temp_filters[LZMA_FILTERS_MAX + 1];

	size_t i = 0;
	do {
		if (i == LZMA_FILTERS_MAX) {
			errmsg = "The maximum number of filters is four";
			goto error;
		}

		// Skip "--" if present.
		if ((*str)[0] == '-' && (*str)[1] == '-')
			*str += 2;

		// Locate the end of "filter:name1=value1,name2=value2",
		// stopping at the first "--" or a single space.
		const char *filter_end = *str;
		while (filter_end[0] != '\0') {
			if ((filter_end[0] == '-' && filter_end[1] == '-')
					|| filter_end[0] == ' ')
				break;

			++filter_end;
		}

		// Inputs that have "--" at the end or "-- " in the middle
		// will result in an empty filter name.
		if (filter_end == *str) {
			errmsg = "Filter name is missing";
			goto error;
		}

		errmsg = parse_filter(str, filter_end, &temp_filters[i],
				allocator, only_xz);
		if (errmsg != NULL)
			goto error;

		// Skip trailing spaces.
		while (**str == ' ')
			++*str;

		++i;
	} while (**str != '\0');

	// Seems to be good, terminate the array so that
	// basic validation can be done.
	temp_filters[i].id = LZMA_VLI_UNKNOWN;
	temp_filters[i].options = NULL;

	// Do basic validation if the application didn't prohibit it.
	if ((flags & LZMA_STR_NO_VALIDATION) == 0) {
		size_t dummy;
		const lzma_ret ret = lzma_validate_chain(temp_filters, &dummy);
		assert(ret == LZMA_OK || ret == LZMA_OPTIONS_ERROR);
		if (ret != LZMA_OK) {
			errmsg = "Invalid filter chain "
					"('lzma2' missing at the end?)";
			goto error;
		}
	}

	// All good. Copy the filters to the application supplied array.
	memcpy(filters, temp_filters, (i + 1) * sizeof(lzma_filter));
	return NULL;

error:
	// Free the filter options that were successfully decoded.
	while (i-- > 0)
		lzma_free(temp_filters[i].options, allocator);

	return errmsg;
}


extern LZMA_API(const char *)
lzma_str_to_filters(const char *str, int *error_pos, lzma_filter *filters,
		uint32_t flags, const lzma_allocator *allocator)
{
	if (str == NULL || filters == NULL)
		return "Unexpected NULL pointer argument(s) "
				"to lzma_str_to_filters()";

	// Validate the flags.
	const uint32_t supported_flags
			= LZMA_STR_ALL_FILTERS
			| LZMA_STR_NO_VALIDATION;

	if (flags & ~supported_flags)
		return "Unsupported flags to lzma_str_to_filters()";

	const char *used = str;
	const char *errmsg = str_to_filters(&used, filters, flags, allocator);

	if (error_pos != NULL) {
		const size_t n = (size_t)(used - str);
		*error_pos = n > INT_MAX ? INT_MAX : (int)n;
	}

	return errmsg;
}


/// Converts options of one filter to a string.
///
/// The caller must have already put the filter name in the destination
/// string. Since it is possible that no options will be needed, the caller
/// won't have put a delimiter character (':' or '=') in the string yet.
/// We will add it if at least one option will be added to the string.
static void
strfy_filter(lzma_str *dest, const char *delimiter,
		const option_map *optmap, size_t optmap_count,
		const void *filter_options)
{
	for (size_t i = 0; i < optmap_count; ++i) {
		// No attempt is made to reverse LZMA1/2 preset.
		if (optmap[i].type == OPTMAP_TYPE_LZMA_PRESET)
			continue;

		// All options have integer values, some just are mapped
		// to a string with a name_value_map. LZMA1/2 preset
		// isn't reversed back to preset=PRESET form.
		uint32_t v;
		const void *ptr
			= (const char *)filter_options + optmap[i].offset;
		switch (optmap[i].type) {
			case OPTMAP_TYPE_LZMA_MODE:
				v = *(const lzma_mode *)ptr;
				break;

			case OPTMAP_TYPE_LZMA_MATCH_FINDER:
				v = *(const lzma_match_finder *)ptr;
				break;

			default:
				v = *(const uint32_t *)ptr;
				break;
		}

		// Skip this if this option should be omitted from
		// the string when the value is zero.
		if (v == 0 && (optmap[i].flags & OPTMAP_NO_STRFY_ZERO))
			continue;

		// Before the first option we add whatever delimiter
		// the caller gave us. For later options a comma is used.
		str_append_str(dest, delimiter);
		delimiter = ",";

		// Add the option name and equals sign.
		str_append_str(dest, optmap[i].name);
		str_append_str(dest, "=");

		if (optmap[i].flags & OPTMAP_USE_NAME_VALUE_MAP) {
			const name_value_map *map = optmap[i].u.map;
			size_t j = 0;
			while (true) {
				if (map[j].name[0] == '\0') {
					str_append_str(dest, "UNKNOWN");
					break;
				}

				if (map[j].value == v) {
					str_append_str(dest, map[j].name);
					break;
				}

				++j;
			}
		} else {
			str_append_u32(dest, v,
				optmap[i].flags & OPTMAP_USE_BYTE_SUFFIX);
		}
	}

	return;
}


extern LZMA_API(lzma_ret)
lzma_str_from_filters(char **output_str, const lzma_filter *filters,
		uint32_t flags, const lzma_allocator *allocator)
{
	// On error *output_str is always set to NULL.
	// Do it as the very first step.
	if (output_str == NULL)
		return LZMA_PROG_ERROR;

	*output_str = NULL;

	if (filters == NULL)
		return LZMA_PROG_ERROR;

	// Validate the flags.
	const uint32_t supported_flags
			= LZMA_STR_ENCODER
			| LZMA_STR_DECODER
			| LZMA_STR_GETOPT_LONG
			| LZMA_STR_NO_SPACES;

	if (flags & ~supported_flags)
		return LZMA_OPTIONS_ERROR;

	// There must be at least one filter.
	if (filters[0].id == LZMA_VLI_UNKNOWN)
		return LZMA_OPTIONS_ERROR;

	// Allocate memory for the output string.
	lzma_str dest;
	return_if_error(str_init(&dest, allocator));

	const bool show_opts = (flags & (LZMA_STR_ENCODER | LZMA_STR_DECODER));

	const char *opt_delim = (flags & LZMA_STR_GETOPT_LONG) ? "=" : ":";

	for (size_t i = 0; filters[i].id != LZMA_VLI_UNKNOWN; ++i) {
		// If we reach LZMA_FILTERS_MAX, then the filters array
		// is too large since the ID cannot be LZMA_VLI_UNKNOWN here.
		if (i == LZMA_FILTERS_MAX) {
			str_free(&dest, allocator);
			return LZMA_OPTIONS_ERROR;
		}

		// Don't add a space between filters if the caller
		// doesn't want them.
		if (i > 0 && !(flags & LZMA_STR_NO_SPACES))
			str_append_str(&dest, " ");

		// Use dashes for xz getopt_long() compatible syntax but also
		// use dashes to separate filters when spaces weren't wanted.
		if ((flags & LZMA_STR_GETOPT_LONG)
				|| (i > 0 && (flags & LZMA_STR_NO_SPACES)))
			str_append_str(&dest, "--");

		size_t j = 0;
		while (true) {
			if (j == ARRAY_SIZE(filter_name_map)) {
				// Filter ID in filters[i].id isn't supported.
				str_free(&dest, allocator);
				return LZMA_OPTIONS_ERROR;
			}

			if (filter_name_map[j].id == filters[i].id) {
				// Add the filter name.
				str_append_str(&dest, filter_name_map[j].name);

				// If only the filter names were wanted then
				// skip to the next filter. In this case
				// .options is ignored and may be NULL even
				// when the filter doesn't allow NULL options.
				if (!show_opts)
					break;

				if (filters[i].options == NULL) {
					if (!filter_name_map[j].allow_null) {
						// Filter-specific options
						// are missing but with
						// this filter the options
						// structure is mandatory.
						str_free(&dest, allocator);
						return LZMA_OPTIONS_ERROR;
					}

					// .options is allowed to be NULL.
					// There is no need to add any
					// options to the string.
					break;
				}

				// Options structure is available. Add
				// the filter options to the string.
				const size_t optmap_count
					= (flags & LZMA_STR_ENCODER)
					? filter_name_map[j].strfy_encoder
					: filter_name_map[j].strfy_decoder;
				strfy_filter(&dest, opt_delim,
						filter_name_map[j].optmap,
						optmap_count,
						filters[i].options);
				break;
			}

			++j;
		}
	}

	return str_finish(output_str, &dest, allocator);
}


extern LZMA_API(lzma_ret)
lzma_str_list_filters(char **output_str, lzma_vli filter_id, uint32_t flags,
		const lzma_allocator *allocator)
{
	// On error *output_str is always set to NULL.
	// Do it as the very first step.
	if (output_str == NULL)
		return LZMA_PROG_ERROR;

	*output_str = NULL;

	// Validate the flags.
	const uint32_t supported_flags
			= LZMA_STR_ALL_FILTERS
			| LZMA_STR_ENCODER
			| LZMA_STR_DECODER
			| LZMA_STR_GETOPT_LONG;

	if (flags & ~supported_flags)
		return LZMA_OPTIONS_ERROR;

	// Allocate memory for the output string.
	lzma_str dest;
	return_if_error(str_init(&dest, allocator));

	// If only listing the filter names then separate them with spaces.
	// Otherwise use newlines.
	const bool show_opts = (flags & (LZMA_STR_ENCODER | LZMA_STR_DECODER));
	const char *filter_delim = show_opts ? "\n" : " ";

	const char *opt_delim = (flags & LZMA_STR_GETOPT_LONG) ? "=" : ":";
	bool first_filter_printed = false;

	for (size_t i = 0; i < ARRAY_SIZE(filter_name_map); ++i) {
		// If we are printing only one filter then skip others.
		if (filter_id != LZMA_VLI_UNKNOWN
				&& filter_id != filter_name_map[i].id)
			continue;

		// If we are printing only .xz filters then skip the others.
		if (filter_name_map[i].id >= LZMA_FILTER_RESERVED_START
				&& (flags & LZMA_STR_ALL_FILTERS) == 0
				&& filter_id == LZMA_VLI_UNKNOWN)
			continue;

		// Add a new line if this isn't the first filter being
		// written to the string.
		if (first_filter_printed)
			str_append_str(&dest, filter_delim);

		first_filter_printed = true;

		if (flags & LZMA_STR_GETOPT_LONG)
			str_append_str(&dest, "--");

		str_append_str(&dest, filter_name_map[i].name);

		// If only the filter names were wanted then continue
		// to the next filter.
		if (!show_opts)
			continue;

		const option_map *optmap = filter_name_map[i].optmap;
		const char *d = opt_delim;

		const size_t end = (flags & LZMA_STR_ENCODER)
				? filter_name_map[i].strfy_encoder
				: filter_name_map[i].strfy_decoder;

		for (size_t j = 0; j < end; ++j) {
			// The first option is delimited from the filter
			// name using "=" or ":" and the rest of the options
			// are separated with ",".
			str_append_str(&dest, d);
			d = ",";

			// optname=<possible_values>
			str_append_str(&dest, optmap[j].name);
			str_append_str(&dest, "=<");

			if (optmap[j].type == OPTMAP_TYPE_LZMA_PRESET) {
				// LZMA1/2 preset has its custom help string.
				str_append_str(&dest, LZMA12_PRESET_STR);
			} else if (optmap[j].flags
					& OPTMAP_USE_NAME_VALUE_MAP) {
				// Separate the possible option values by "|".
				const name_value_map *m = optmap[j].u.map;
				for (size_t k = 0; m[k].name[0] != '\0'; ++k) {
					if (k > 0)
						str_append_str(&dest, "|");

					str_append_str(&dest, m[k].name);
				}
			} else {
				// Integer range is shown as min-max.
				const bool use_byte_suffix = optmap[j].flags
						& OPTMAP_USE_BYTE_SUFFIX;
				str_append_u32(&dest, optmap[j].u.range.min,
						use_byte_suffix);
				str_append_str(&dest, "-");
				str_append_u32(&dest, optmap[j].u.range.max,
						use_byte_suffix);
			}

			str_append_str(&dest, ">");
		}
	}

	// If no filters were added to the string then it must be because
	// the caller provided an unsupported Filter ID.
	if (!first_filter_printed) {
		str_free(&dest, allocator);
		return LZMA_OPTIONS_ERROR;
	}

	return str_finish(output_str, &dest, allocator);
}
