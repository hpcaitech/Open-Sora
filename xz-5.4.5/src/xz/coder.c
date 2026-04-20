///////////////////////////////////////////////////////////////////////////////
//
/// \file       coder.c
/// \brief      Compresses or uncompresses a file
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "private.h"


/// Return value type for coder_init().
enum coder_init_ret {
	CODER_INIT_NORMAL,
	CODER_INIT_PASSTHRU,
	CODER_INIT_ERROR,
};


enum operation_mode opt_mode = MODE_COMPRESS;
enum format_type opt_format = FORMAT_AUTO;
bool opt_auto_adjust = true;
bool opt_single_stream = false;
uint64_t opt_block_size = 0;
uint64_t *opt_block_list = NULL;


/// Stream used to communicate with liblzma
static lzma_stream strm = LZMA_STREAM_INIT;

/// Filters needed for all encoding all formats, and also decoding in raw data
static lzma_filter filters[LZMA_FILTERS_MAX + 1];

/// Input and output buffers
static io_buf in_buf;
static io_buf out_buf;

/// Number of filters. Zero indicates that we are using a preset.
static uint32_t filters_count = 0;

/// Number of the preset (0-9)
static uint32_t preset_number = LZMA_PRESET_DEFAULT;

/// Integrity check type
static lzma_check check;

/// This becomes false if the --check=CHECK option is used.
static bool check_default = true;

/// Indicates if unconsumed input is allowed to remain after
/// decoding has successfully finished. This is set for each file
/// in coder_init().
static bool allow_trailing_input;

#ifdef MYTHREAD_ENABLED
static lzma_mt mt_options = {
	.flags = 0,
	.timeout = 300,
	.filters = filters,
};
#endif


extern void
coder_set_check(lzma_check new_check)
{
	check = new_check;
	check_default = false;
	return;
}


static void
forget_filter_chain(void)
{
	// Setting a preset makes us forget a possibly defined custom
	// filter chain.
	while (filters_count > 0) {
		--filters_count;
		free(filters[filters_count].options);
		filters[filters_count].options = NULL;
	}

	return;
}


extern void
coder_set_preset(uint32_t new_preset)
{
	preset_number &= ~LZMA_PRESET_LEVEL_MASK;
	preset_number |= new_preset;
	forget_filter_chain();
	return;
}


extern void
coder_set_extreme(void)
{
	preset_number |= LZMA_PRESET_EXTREME;
	forget_filter_chain();
	return;
}


extern void
coder_add_filter(lzma_vli id, void *options)
{
	if (filters_count == LZMA_FILTERS_MAX)
		message_fatal(_("Maximum number of filters is four"));

	filters[filters_count].id = id;
	filters[filters_count].options = options;
	++filters_count;

	// Setting a custom filter chain makes us forget the preset options.
	// This makes a difference if one specifies e.g. "xz -9 --lzma2 -e"
	// where the custom filter chain resets the preset level back to
	// the default 6, making the example equivalent to "xz -6e".
	preset_number = LZMA_PRESET_DEFAULT;

	return;
}


tuklib_attr_noreturn
static void
memlimit_too_small(uint64_t memory_usage)
{
	message(V_ERROR, _("Memory usage limit is too low for the given "
			"filter setup."));
	message_mem_needed(V_ERROR, memory_usage);
	tuklib_exit(E_ERROR, E_ERROR, false);
}


extern void
coder_set_compression_settings(void)
{
#ifdef HAVE_LZIP_DECODER
	// .lz compression isn't supported.
	assert(opt_format != FORMAT_LZIP);
#endif

	// The default check type is CRC64, but fallback to CRC32
	// if CRC64 isn't supported by the copy of liblzma we are
	// using. CRC32 is always supported.
	if (check_default) {
		check = LZMA_CHECK_CRC64;
		if (!lzma_check_is_supported(check))
			check = LZMA_CHECK_CRC32;
	}

	// Options for LZMA1 or LZMA2 in case we are using a preset.
	static lzma_options_lzma opt_lzma;

	if (filters_count == 0) {
		// We are using a preset. This is not a good idea in raw mode
		// except when playing around with things. Different versions
		// of this software may use different options in presets, and
		// thus make uncompressing the raw data difficult.
		if (opt_format == FORMAT_RAW) {
			// The message is shown only if warnings are allowed
			// but the exit status isn't changed.
			message(V_WARNING, _("Using a preset in raw mode "
					"is discouraged."));
			message(V_WARNING, _("The exact options of the "
					"presets may vary between software "
					"versions."));
		}

		// Get the preset for LZMA1 or LZMA2.
		if (lzma_lzma_preset(&opt_lzma, preset_number))
			message_bug();

		// Use LZMA2 except with --format=lzma we use LZMA1.
		filters[0].id = opt_format == FORMAT_LZMA
				? LZMA_FILTER_LZMA1 : LZMA_FILTER_LZMA2;
		filters[0].options = &opt_lzma;
		filters_count = 1;
	}

	// Terminate the filter options array.
	filters[filters_count].id = LZMA_VLI_UNKNOWN;

	// If we are using the .lzma format, allow exactly one filter
	// which has to be LZMA1.
	if (opt_format == FORMAT_LZMA && (filters_count != 1
			|| filters[0].id != LZMA_FILTER_LZMA1))
		message_fatal(_("The .lzma format supports only "
				"the LZMA1 filter"));

	// If we are using the .xz format, make sure that there is no LZMA1
	// filter to prevent LZMA_PROG_ERROR.
	if (opt_format == FORMAT_XZ)
		for (size_t i = 0; i < filters_count; ++i)
			if (filters[i].id == LZMA_FILTER_LZMA1)
				message_fatal(_("LZMA1 cannot be used "
						"with the .xz format"));

	// Print the selected filter chain.
	message_filters_show(V_DEBUG, filters);

	// The --flush-timeout option requires LZMA_SYNC_FLUSH support
	// from the filter chain. Currently threaded encoder doesn't support
	// LZMA_SYNC_FLUSH so single-threaded mode must be used.
	if (opt_mode == MODE_COMPRESS && opt_flush_timeout != 0) {
		for (size_t i = 0; i < filters_count; ++i) {
			switch (filters[i].id) {
			case LZMA_FILTER_LZMA2:
			case LZMA_FILTER_DELTA:
				break;

			default:
				message_fatal(_("The filter chain is "
					"incompatible with --flush-timeout"));
			}
		}

		if (hardware_threads_is_mt()) {
			message(V_WARNING, _("Switching to single-threaded "
					"mode due to --flush-timeout"));
			hardware_threads_set(1);
		}
	}

	// Get the memory usage. Note that if --format=raw was used,
	// we can be decompressing.
	//
	// If multithreaded .xz compression is done, this value will be
	// replaced.
	uint64_t memory_limit = hardware_memlimit_get(opt_mode);
	uint64_t memory_usage = UINT64_MAX;
	if (opt_mode == MODE_COMPRESS) {
#ifdef HAVE_ENCODERS
#	ifdef MYTHREAD_ENABLED
		if (opt_format == FORMAT_XZ && hardware_threads_is_mt()) {
			memory_limit = hardware_memlimit_mtenc_get();
			mt_options.threads = hardware_threads_get();
			mt_options.block_size = opt_block_size;
			mt_options.check = check;
			memory_usage = lzma_stream_encoder_mt_memusage(
					&mt_options);
			if (memory_usage != UINT64_MAX)
				message(V_DEBUG, _("Using up to %" PRIu32
						" threads."),
						mt_options.threads);
		} else
#	endif
		{
			memory_usage = lzma_raw_encoder_memusage(filters);
		}
#endif
	} else {
#ifdef HAVE_DECODERS
		memory_usage = lzma_raw_decoder_memusage(filters);
#endif
	}

	if (memory_usage == UINT64_MAX)
		message_fatal(_("Unsupported filter chain or filter options"));

	// Print memory usage info before possible dictionary
	// size auto-adjusting.
	//
	// NOTE: If only encoder support was built, we cannot show the
	// what the decoder memory usage will be.
	message_mem_needed(V_DEBUG, memory_usage);
#ifdef HAVE_DECODERS
	if (opt_mode == MODE_COMPRESS) {
		const uint64_t decmem = lzma_raw_decoder_memusage(filters);
		if (decmem != UINT64_MAX)
			message(V_DEBUG, _("Decompression will need "
					"%s MiB of memory."), uint64_to_str(
						round_up_to_mib(decmem), 0));
	}
#endif

	if (memory_usage <= memory_limit)
		return;

	// With --format=raw settings are never adjusted to meet
	// the memory usage limit.
	if (opt_format == FORMAT_RAW)
		memlimit_too_small(memory_usage);

	assert(opt_mode == MODE_COMPRESS);

#ifdef HAVE_ENCODERS
#	ifdef MYTHREAD_ENABLED
	if (opt_format == FORMAT_XZ && hardware_threads_is_mt()) {
		// Try to reduce the number of threads before
		// adjusting the compression settings down.
		while (mt_options.threads > 1) {
			// Reduce the number of threads by one and check
			// the memory usage.
			--mt_options.threads;
			memory_usage = lzma_stream_encoder_mt_memusage(
					&mt_options);
			if (memory_usage == UINT64_MAX)
				message_bug();

			if (memory_usage <= memory_limit) {
				// The memory usage is now low enough.
				message(V_WARNING, _("Reduced the number of "
					"threads from %s to %s to not exceed "
					"the memory usage limit of %s MiB"),
					uint64_to_str(
						hardware_threads_get(), 0),
					uint64_to_str(mt_options.threads, 1),
					uint64_to_str(round_up_to_mib(
						memory_limit), 2));
				return;
			}
		}

		// If the memory usage limit is only a soft limit (automatic
		// number of threads and no --memlimit-compress), the limit
		// is only used to reduce the number of threads and once at
		// just one thread, the limit is completely ignored. This
		// way -T0 won't use insane amount of memory but at the same
		// time the soft limit will never make xz fail and never make
		// xz change settings that would affect the compressed output.
		if (hardware_memlimit_mtenc_is_default()) {
			message(V_WARNING, _("Reduced the number of threads "
				"from %s to one. The automatic memory usage "
				"limit of %s MiB is still being exceeded. "
				"%s MiB of memory is required. "
				"Continuing anyway."),
				uint64_to_str(hardware_threads_get(), 0),
				uint64_to_str(
					round_up_to_mib(memory_limit), 1),
				uint64_to_str(
					round_up_to_mib(memory_usage), 2));
			return;
		}

		// If --no-adjust was used, we cannot drop to single-threaded
		// mode since it produces different compressed output.
		//
		// NOTE: In xz 5.2.x, --no-adjust also prevented reducing
		// the number of threads. This changed in 5.3.3alpha.
		if (!opt_auto_adjust)
			memlimit_too_small(memory_usage);

		// Switch to single-threaded mode. It uses
		// less memory than using one thread in
		// the multithreaded mode but the output
		// is also different.
		hardware_threads_set(1);
		memory_usage = lzma_raw_encoder_memusage(filters);
		message(V_WARNING, _("Switching to single-threaded mode "
			"to not exceed the memory usage limit of %s MiB"),
			uint64_to_str(round_up_to_mib(memory_limit), 0));
	}
#	endif

	if (memory_usage <= memory_limit)
		return;

	// Don't adjust LZMA2 or LZMA1 dictionary size if --no-adjust
	// was specified as that would change the compressed output.
	if (!opt_auto_adjust)
		memlimit_too_small(memory_usage);

	// Look for the last filter if it is LZMA2 or LZMA1, so we can make
	// it use less RAM. With other filters we don't know what to do.
	size_t i = 0;
	while (filters[i].id != LZMA_FILTER_LZMA2
			&& filters[i].id != LZMA_FILTER_LZMA1) {
		if (filters[i].id == LZMA_VLI_UNKNOWN)
			memlimit_too_small(memory_usage);

		++i;
	}

	// Decrease the dictionary size until we meet the memory
	// usage limit. First round down to full mebibytes.
	lzma_options_lzma *opt = filters[i].options;
	const uint32_t orig_dict_size = opt->dict_size;
	opt->dict_size &= ~((UINT32_C(1) << 20) - 1);
	while (true) {
		// If it is below 1 MiB, auto-adjusting failed. We could be
		// more sophisticated and scale it down even more, but let's
		// see if many complain about this version.
		//
		// FIXME: Displays the scaled memory usage instead
		// of the original.
		if (opt->dict_size < (UINT32_C(1) << 20))
			memlimit_too_small(memory_usage);

		memory_usage = lzma_raw_encoder_memusage(filters);
		if (memory_usage == UINT64_MAX)
			message_bug();

		// Accept it if it is low enough.
		if (memory_usage <= memory_limit)
			break;

		// Otherwise 1 MiB down and try again. I hope this
		// isn't too slow method for cases where the original
		// dict_size is very big.
		opt->dict_size -= UINT32_C(1) << 20;
	}

	// Tell the user that we decreased the dictionary size.
	message(V_WARNING, _("Adjusted LZMA%c dictionary size "
			"from %s MiB to %s MiB to not exceed "
			"the memory usage limit of %s MiB"),
			filters[i].id == LZMA_FILTER_LZMA2
				? '2' : '1',
			uint64_to_str(orig_dict_size >> 20, 0),
			uint64_to_str(opt->dict_size >> 20, 1),
			uint64_to_str(round_up_to_mib(memory_limit), 2));
#endif

	return;
}


#ifdef HAVE_DECODERS
/// Return true if the data in in_buf seems to be in the .xz format.
static bool
is_format_xz(void)
{
	// Specify the magic as hex to be compatible with EBCDIC systems.
	static const uint8_t magic[6] = { 0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00 };
	return strm.avail_in >= sizeof(magic)
			&& memcmp(in_buf.u8, magic, sizeof(magic)) == 0;
}


/// Return true if the data in in_buf seems to be in the .lzma format.
static bool
is_format_lzma(void)
{
	// The .lzma header is 13 bytes.
	if (strm.avail_in < 13)
		return false;

	// Decode the LZMA1 properties.
	lzma_filter filter = { .id = LZMA_FILTER_LZMA1 };
	if (lzma_properties_decode(&filter, NULL, in_buf.u8, 5) != LZMA_OK)
		return false;

	// A hack to ditch tons of false positives: We allow only dictionary
	// sizes that are 2^n or 2^n + 2^(n-1) or UINT32_MAX. LZMA_Alone
	// created only files with 2^n, but accepts any dictionary size.
	// If someone complains, this will be reconsidered.
	lzma_options_lzma *opt = filter.options;
	const uint32_t dict_size = opt->dict_size;
	free(opt);

	if (dict_size != UINT32_MAX) {
		uint32_t d = dict_size - 1;
		d |= d >> 2;
		d |= d >> 3;
		d |= d >> 4;
		d |= d >> 8;
		d |= d >> 16;
		++d;
		if (d != dict_size || dict_size == 0)
			return false;
	}

	// Another hack to ditch false positives: Assume that if the
	// uncompressed size is known, it must be less than 256 GiB.
	// Again, if someone complains, this will be reconsidered.
	uint64_t uncompressed_size = 0;
	for (size_t i = 0; i < 8; ++i)
		uncompressed_size |= (uint64_t)(in_buf.u8[5 + i]) << (i * 8);

	if (uncompressed_size != UINT64_MAX
			&& uncompressed_size > (UINT64_C(1) << 38))
		return false;

	return true;
}


#ifdef HAVE_LZIP_DECODER
/// Return true if the data in in_buf seems to be in the .lz format.
static bool
is_format_lzip(void)
{
	static const uint8_t magic[4] = { 0x4C, 0x5A, 0x49, 0x50 };
	return strm.avail_in >= sizeof(magic)
			&& memcmp(in_buf.u8, magic, sizeof(magic)) == 0;
}
#endif
#endif


/// Detect the input file type (for now, this done only when decompressing),
/// and initialize an appropriate coder. Return value indicates if a normal
/// liblzma-based coder was initialized (CODER_INIT_NORMAL), if passthru
/// mode should be used (CODER_INIT_PASSTHRU), or if an error occurred
/// (CODER_INIT_ERROR).
static enum coder_init_ret
coder_init(file_pair *pair)
{
	lzma_ret ret = LZMA_PROG_ERROR;

	// In most cases if there is input left when coding finishes,
	// something has gone wrong. Exceptions are --single-stream
	// and decoding .lz files which can contain trailing non-.lz data.
	// These will be handled later in this function.
	allow_trailing_input = false;

	if (opt_mode == MODE_COMPRESS) {
#ifdef HAVE_ENCODERS
		switch (opt_format) {
		case FORMAT_AUTO:
			// args.c ensures this.
			assert(0);
			break;

		case FORMAT_XZ:
#	ifdef MYTHREAD_ENABLED
			if (hardware_threads_is_mt())
				ret = lzma_stream_encoder_mt(
						&strm, &mt_options);
			else
#	endif
				ret = lzma_stream_encoder(
						&strm, filters, check);
			break;

		case FORMAT_LZMA:
			ret = lzma_alone_encoder(&strm, filters[0].options);
			break;

#	ifdef HAVE_LZIP_DECODER
		case FORMAT_LZIP:
			// args.c should disallow this.
			assert(0);
			ret = LZMA_PROG_ERROR;
			break;
#	endif

		case FORMAT_RAW:
			ret = lzma_raw_encoder(&strm, filters);
			break;
		}
#endif
	} else {
#ifdef HAVE_DECODERS
		uint32_t flags = 0;

		// It seems silly to warn about unsupported check if the
		// check won't be verified anyway due to --ignore-check.
		if (opt_ignore_check)
			flags |= LZMA_IGNORE_CHECK;
		else
			flags |= LZMA_TELL_UNSUPPORTED_CHECK;

		if (opt_single_stream)
			allow_trailing_input = true;
		else
			flags |= LZMA_CONCATENATED;

		// We abuse FORMAT_AUTO to indicate unknown file format,
		// for which we may consider passthru mode.
		enum format_type init_format = FORMAT_AUTO;

		switch (opt_format) {
		case FORMAT_AUTO:
			// .lz is checked before .lzma since .lzma detection
			// is more complicated (no magic bytes).
			if (is_format_xz())
				init_format = FORMAT_XZ;
#	ifdef HAVE_LZIP_DECODER
			else if (is_format_lzip())
				init_format = FORMAT_LZIP;
#	endif
			else if (is_format_lzma())
				init_format = FORMAT_LZMA;
			break;

		case FORMAT_XZ:
			if (is_format_xz())
				init_format = FORMAT_XZ;
			break;

		case FORMAT_LZMA:
			if (is_format_lzma())
				init_format = FORMAT_LZMA;
			break;

#	ifdef HAVE_LZIP_DECODER
		case FORMAT_LZIP:
			if (is_format_lzip())
				init_format = FORMAT_LZIP;
			break;
#	endif

		case FORMAT_RAW:
			init_format = FORMAT_RAW;
			break;
		}

		switch (init_format) {
		case FORMAT_AUTO:
			// Unknown file format. If --decompress --stdout
			// --force have been given, then we copy the input
			// as is to stdout. Checking for MODE_DECOMPRESS
			// is needed, because we don't want to do use
			// passthru mode with --test.
			if (opt_mode == MODE_DECOMPRESS
					&& opt_stdout && opt_force) {
				// These are needed for progress info.
				strm.total_in = 0;
				strm.total_out = 0;
				return CODER_INIT_PASSTHRU;
			}

			ret = LZMA_FORMAT_ERROR;
			break;

		case FORMAT_XZ:
#	ifdef MYTHREAD_ENABLED
			mt_options.flags = flags;

			mt_options.threads = hardware_threads_get();
			mt_options.memlimit_stop
				= hardware_memlimit_get(MODE_DECOMPRESS);

			// If single-threaded mode was requested, set the
			// memlimit for threading to zero. This forces the
			// decoder to use single-threaded mode which matches
			// the behavior of lzma_stream_decoder().
			//
			// Otherwise use the limit for threaded decompression
			// which has a sane default (users are still free to
			// make it insanely high though).
			mt_options.memlimit_threading
					= mt_options.threads == 1
					? 0 : hardware_memlimit_mtdec_get();

			ret = lzma_stream_decoder_mt(&strm, &mt_options);
#	else
			ret = lzma_stream_decoder(&strm,
					hardware_memlimit_get(
						MODE_DECOMPRESS), flags);
#	endif
			break;

		case FORMAT_LZMA:
			ret = lzma_alone_decoder(&strm,
					hardware_memlimit_get(
						MODE_DECOMPRESS));
			break;

#	ifdef HAVE_LZIP_DECODER
		case FORMAT_LZIP:
			allow_trailing_input = true;
			ret = lzma_lzip_decoder(&strm,
					hardware_memlimit_get(
						MODE_DECOMPRESS), flags);
			break;
#	endif

		case FORMAT_RAW:
			// Memory usage has already been checked in
			// coder_set_compression_settings().
			ret = lzma_raw_decoder(&strm, filters);
			break;
		}

		// Try to decode the headers. This will catch too low
		// memory usage limit in case it happens in the first
		// Block of the first Stream, which is where it very
		// probably will happen if it is going to happen.
		//
		// This will also catch unsupported check type which
		// we treat as a warning only. If there are empty
		// concatenated Streams with unsupported check type then
		// the message can be shown more than once here. The loop
		// is used in case there is first a warning about
		// unsupported check type and then the first Block
		// would exceed the memlimit.
		if (ret == LZMA_OK && init_format != FORMAT_RAW) {
			strm.next_out = NULL;
			strm.avail_out = 0;
			while ((ret = lzma_code(&strm, LZMA_RUN))
					== LZMA_UNSUPPORTED_CHECK)
				message_warning(_("%s: %s"), pair->src_name,
						message_strm(ret));

			// With --single-stream lzma_code won't wait for
			// LZMA_FINISH and thus it can return LZMA_STREAM_END
			// if the file has no uncompressed data inside.
			// So treat LZMA_STREAM_END as LZMA_OK here.
			// When lzma_code() is called again in coder_normal()
			// it will return LZMA_STREAM_END again.
			if (ret == LZMA_STREAM_END)
				ret = LZMA_OK;
		}
#endif
	}

	if (ret != LZMA_OK) {
		message_error(_("%s: %s"), pair->src_name, message_strm(ret));
		if (ret == LZMA_MEMLIMIT_ERROR)
			message_mem_needed(V_ERROR, lzma_memusage(&strm));

		return CODER_INIT_ERROR;
	}

	return CODER_INIT_NORMAL;
}


/// Resolve conflicts between opt_block_size and opt_block_list in single
/// threaded mode. We want to default to opt_block_list, except when it is
/// larger than opt_block_size. If this is the case for the current Block
/// at *list_pos, then we break into smaller Blocks. Otherwise advance
/// to the next Block in opt_block_list, and break apart if needed.
static void
split_block(uint64_t *block_remaining,
	    uint64_t *next_block_remaining,
	    size_t *list_pos)
{
	if (*next_block_remaining > 0) {
		// The Block at *list_pos has previously been split up.
		assert(!hardware_threads_is_mt());
		assert(opt_block_size > 0);
		assert(opt_block_list != NULL);

		if (*next_block_remaining > opt_block_size) {
			// We have to split the current Block at *list_pos
			// into another opt_block_size length Block.
			*block_remaining = opt_block_size;
		} else {
			// This is the last remaining split Block for the
			// Block at *list_pos.
			*block_remaining = *next_block_remaining;
		}

		*next_block_remaining -= *block_remaining;

	} else {
		// The Block at *list_pos has been finished. Go to the next
		// entry in the list. If the end of the list has been reached,
		// reuse the size of the last Block.
		if (opt_block_list[*list_pos + 1] != 0)
			++*list_pos;

		*block_remaining = opt_block_list[*list_pos];

		// If in single-threaded mode, split up the Block if needed.
		// This is not needed in multi-threaded mode because liblzma
		// will do this due to how threaded encoding works.
		if (!hardware_threads_is_mt() && opt_block_size > 0
				&& *block_remaining > opt_block_size) {
			*next_block_remaining
					= *block_remaining - opt_block_size;
			*block_remaining = opt_block_size;
		}
	}
}


static bool
coder_write_output(file_pair *pair)
{
	if (opt_mode != MODE_TEST) {
		if (io_write(pair, &out_buf, IO_BUFFER_SIZE - strm.avail_out))
			return true;
	}

	strm.next_out = out_buf.u8;
	strm.avail_out = IO_BUFFER_SIZE;
	return false;
}


/// Compress or decompress using liblzma.
static bool
coder_normal(file_pair *pair)
{
	// Encoder needs to know when we have given all the input to it.
	// The decoders need to know it too when we are using
	// LZMA_CONCATENATED. We need to check for src_eof here, because
	// the first input chunk has been already read if decompressing,
	// and that may have been the only chunk we will read.
	lzma_action action = pair->src_eof ? LZMA_FINISH : LZMA_RUN;

	lzma_ret ret;

	// Assume that something goes wrong.
	bool success = false;

	// block_remaining indicates how many input bytes to encode before
	// finishing the current .xz Block. The Block size is set with
	// --block-size=SIZE and --block-list. They have an effect only when
	// compressing to the .xz format. If block_remaining == UINT64_MAX,
	// only a single block is created.
	uint64_t block_remaining = UINT64_MAX;

	// next_block_remaining for when we are in single-threaded mode and
	// the Block in --block-list is larger than the --block-size=SIZE.
	uint64_t next_block_remaining = 0;

	// Position in opt_block_list. Unused if --block-list wasn't used.
	size_t list_pos = 0;

	// Handle --block-size for single-threaded mode and the first step
	// of --block-list.
	if (opt_mode == MODE_COMPRESS && opt_format == FORMAT_XZ) {
		// --block-size doesn't do anything here in threaded mode,
		// because the threaded encoder will take care of splitting
		// to fixed-sized Blocks.
		if (!hardware_threads_is_mt() && opt_block_size > 0)
			block_remaining = opt_block_size;

		// If --block-list was used, start with the first size.
		//
		// For threaded case, --block-size specifies how big Blocks
		// the encoder needs to be prepared to create at maximum
		// and --block-list will simultaneously cause new Blocks
		// to be started at specified intervals. To keep things
		// logical, the same is done in single-threaded mode. The
		// output is still not identical because in single-threaded
		// mode the size info isn't written into Block Headers.
		if (opt_block_list != NULL) {
			if (block_remaining < opt_block_list[list_pos]) {
				assert(!hardware_threads_is_mt());
				next_block_remaining = opt_block_list[list_pos]
						- block_remaining;
			} else {
				block_remaining = opt_block_list[list_pos];
			}
		}
	}

	strm.next_out = out_buf.u8;
	strm.avail_out = IO_BUFFER_SIZE;

	while (!user_abort) {
		// Fill the input buffer if it is empty and we aren't
		// flushing or finishing.
		if (strm.avail_in == 0 && action == LZMA_RUN) {
			strm.next_in = in_buf.u8;
			strm.avail_in = io_read(pair, &in_buf,
					my_min(block_remaining,
						IO_BUFFER_SIZE));

			if (strm.avail_in == SIZE_MAX)
				break;

			if (pair->src_eof) {
				action = LZMA_FINISH;

			} else if (block_remaining != UINT64_MAX) {
				// Start a new Block after every
				// opt_block_size bytes of input.
				block_remaining -= strm.avail_in;
				if (block_remaining == 0)
					action = LZMA_FULL_BARRIER;
			}

			if (action == LZMA_RUN && pair->flush_needed)
				action = LZMA_SYNC_FLUSH;
		}

		// Let liblzma do the actual work.
		ret = lzma_code(&strm, action);

		// Write out if the output buffer became full.
		if (strm.avail_out == 0) {
			if (coder_write_output(pair))
				break;
		}

		if (ret == LZMA_STREAM_END && (action == LZMA_SYNC_FLUSH
				|| action == LZMA_FULL_BARRIER)) {
			if (action == LZMA_SYNC_FLUSH) {
				// Flushing completed. Write the pending data
				// out immediately so that the reading side
				// can decompress everything compressed so far.
				if (coder_write_output(pair))
					break;

				// Mark that we haven't seen any new input
				// since the previous flush.
				pair->src_has_seen_input = false;
				pair->flush_needed = false;
			} else {
				// Start a new Block after LZMA_FULL_BARRIER.
				if (opt_block_list == NULL) {
					assert(!hardware_threads_is_mt());
					assert(opt_block_size > 0);
					block_remaining = opt_block_size;
				} else {
					split_block(&block_remaining,
							&next_block_remaining,
							&list_pos);
				}
			}

			// Start a new Block after LZMA_FULL_FLUSH or continue
			// the same block after LZMA_SYNC_FLUSH.
			action = LZMA_RUN;

		} else if (ret != LZMA_OK) {
			// Determine if the return value indicates that we
			// won't continue coding. LZMA_NO_CHECK would be
			// here too if LZMA_TELL_ANY_CHECK was used.
			const bool stop = ret != LZMA_UNSUPPORTED_CHECK;

			if (stop) {
				// Write the remaining bytes even if something
				// went wrong, because that way the user gets
				// as much data as possible, which can be good
				// when trying to get at least some useful
				// data out of damaged files.
				if (coder_write_output(pair))
					break;
			}

			if (ret == LZMA_STREAM_END) {
				if (allow_trailing_input) {
					io_fix_src_pos(pair, strm.avail_in);
					success = true;
					break;
				}

				// Check that there is no trailing garbage.
				// This is needed for LZMA_Alone and raw
				// streams. This is *not* done with .lz files
				// as that format specifically requires
				// allowing trailing garbage.
				if (strm.avail_in == 0 && !pair->src_eof) {
					// Try reading one more byte.
					// Hopefully we don't get any more
					// input, and thus pair->src_eof
					// becomes true.
					strm.avail_in = io_read(
							pair, &in_buf, 1);
					if (strm.avail_in == SIZE_MAX)
						break;

					assert(strm.avail_in == 0
							|| strm.avail_in == 1);
				}

				if (strm.avail_in == 0) {
					assert(pair->src_eof);
					success = true;
					break;
				}

				// We hadn't reached the end of the file.
				ret = LZMA_DATA_ERROR;
				assert(stop);
			}

			// If we get here and stop is true, something went
			// wrong and we print an error. Otherwise it's just
			// a warning and coding can continue.
			if (stop) {
				message_error(_("%s: %s"), pair->src_name,
						message_strm(ret));
			} else {
				message_warning(_("%s: %s"), pair->src_name,
						message_strm(ret));

				// When compressing, all possible errors set
				// stop to true.
				assert(opt_mode != MODE_COMPRESS);
			}

			if (ret == LZMA_MEMLIMIT_ERROR) {
				// Display how much memory it would have
				// actually needed.
				message_mem_needed(V_ERROR,
						lzma_memusage(&strm));
			}

			if (stop)
				break;
		}

		// Show progress information under certain conditions.
		message_progress_update();
	}

	return success;
}


/// Copy from input file to output file without processing the data in any
/// way. This is used only when trying to decompress unrecognized files
/// with --decompress --stdout --force, so the output is always stdout.
static bool
coder_passthru(file_pair *pair)
{
	while (strm.avail_in != 0) {
		if (user_abort)
			return false;

		if (io_write(pair, &in_buf, strm.avail_in))
			return false;

		strm.total_in += strm.avail_in;
		strm.total_out = strm.total_in;
		message_progress_update();

		strm.avail_in = io_read(pair, &in_buf, IO_BUFFER_SIZE);
		if (strm.avail_in == SIZE_MAX)
			return false;
	}

	return true;
}


extern void
coder_run(const char *filename)
{
	// Set and possibly print the filename for the progress message.
	message_filename(filename);

	// Try to open the input file.
	file_pair *pair = io_open_src(filename);
	if (pair == NULL)
		return;

	// Assume that something goes wrong.
	bool success = false;

	if (opt_mode == MODE_COMPRESS) {
		strm.next_in = NULL;
		strm.avail_in = 0;
	} else {
		// Read the first chunk of input data. This is needed
		// to detect the input file type.
		strm.next_in = in_buf.u8;
		strm.avail_in = io_read(pair, &in_buf, IO_BUFFER_SIZE);
	}

	if (strm.avail_in != SIZE_MAX) {
		// Initialize the coder. This will detect the file format
		// and, in decompression or testing mode, check the memory
		// usage of the first Block too. This way we don't try to
		// open the destination file if we see that coding wouldn't
		// work at all anyway. This also avoids deleting the old
		// "target" file if --force was used.
		const enum coder_init_ret init_ret = coder_init(pair);

		if (init_ret != CODER_INIT_ERROR && !user_abort) {
			// Don't open the destination file when --test
			// is used.
			if (opt_mode == MODE_TEST || !io_open_dest(pair)) {
				// Remember the current time. It is needed
				// for progress indicator.
				mytime_set_start_time();

				// Initialize the progress indicator.
				//
				// NOTE: When reading from stdin, fstat()
				// isn't called on it and thus src_st.st_size
				// is zero. If stdin pointed to a regular
				// file, it would still be possible to know
				// the file size but then we would also need
				// to take into account the current reading
				// position since with stdin it isn't
				// necessarily at the beginning of the file.
				const bool is_passthru = init_ret
						== CODER_INIT_PASSTHRU;
				const uint64_t in_size
					= pair->src_st.st_size <= 0
					? 0 : (uint64_t)(pair->src_st.st_size);
				message_progress_start(&strm,
						is_passthru, in_size);

				// Do the actual coding or passthru.
				if (is_passthru)
					success = coder_passthru(pair);
				else
					success = coder_normal(pair);

				message_progress_end(success);
			}
		}
	}

	// Close the file pair. It needs to know if coding was successful to
	// know if the source or target file should be unlinked.
	io_close(pair, success);

	return;
}


#ifndef NDEBUG
extern void
coder_free(void)
{
	lzma_end(&strm);
	return;
}
#endif
