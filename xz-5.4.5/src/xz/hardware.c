///////////////////////////////////////////////////////////////////////////////
//
/// \file       hardware.c
/// \brief      Detection of available hardware resources
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "private.h"


/// Maximum number of worker threads. This can be set with
/// the --threads=NUM command line option.
static uint32_t threads_max = 1;

/// True when the number of threads is automatically determined based
/// on the available hardware threads.
static bool threads_are_automatic = false;

/// If true, then try to use multi-threaded mode (if memlimit allows)
/// even if only one thread was requested explicitly (-T+1).
static bool use_mt_mode_with_one_thread = false;

/// Memory usage limit for compression
static uint64_t memlimit_compress = 0;

/// Memory usage limit for decompression
static uint64_t memlimit_decompress = 0;

/// Default memory usage for multithreaded modes:
///
///   - Default value for --memlimit-compress when automatic number of threads
///     is used. However, if the limit wouldn't allow even one thread then
///     the limit is ignored in coder.c and one thread will be used anyway.
///     This mess is a compromise: we wish to prevent -T0 from using too
///     many threads but we also don't want xz to give an error due to
///     a memlimit that the user didn't explicitly set.
///
///   - Default value for --memlimit-mt-decompress
///
/// This value is calculated in hardware_init() and cannot be changed later.
static uint64_t memlimit_mt_default;

/// Memory usage limit for multithreaded decompression. This is a soft limit:
/// if reducing the number of threads to one isn't enough to keep memory
/// usage below this limit, then one thread is used and this limit is ignored.
/// memlimit_decompress is still obeyed.
///
/// This can be set with --memlimit-mt-decompress. The default value for
/// this is memlimit_mt_default.
static uint64_t memlimit_mtdec;

/// Total amount of physical RAM
static uint64_t total_ram;


extern void
hardware_threads_set(uint32_t n)
{
	// Reset these to false first and set them to true when appropriate.
	threads_are_automatic = false;
	use_mt_mode_with_one_thread = false;

	if (n == 0) {
		// Automatic number of threads was requested.
		// If there is only one hardware thread, multi-threaded
		// mode will still be used if memory limit allows.
		threads_are_automatic = true;
		use_mt_mode_with_one_thread = true;

		// If threading support was enabled at build time,
		// use the number of available CPU cores. Otherwise
		// use one thread since disabling threading support
		// omits lzma_cputhreads() from liblzma.
#ifdef MYTHREAD_ENABLED
		threads_max = lzma_cputhreads();
		if (threads_max == 0)
			threads_max = 1;
#else
		threads_max = 1;
#endif
	} else if (n == UINT32_MAX) {
		use_mt_mode_with_one_thread = true;
		threads_max = 1;
	} else {
		threads_max = n;
	}

	return;
}


extern uint32_t
hardware_threads_get(void)
{
	return threads_max;
}


extern bool
hardware_threads_is_mt(void)
{
#ifdef MYTHREAD_ENABLED
	return threads_max > 1 || use_mt_mode_with_one_thread;
#else
	return false;
#endif
}


extern void
hardware_memlimit_set(uint64_t new_memlimit,
		bool set_compress, bool set_decompress, bool set_mtdec,
		bool is_percentage)
{
	if (is_percentage) {
		assert(new_memlimit > 0);
		assert(new_memlimit <= 100);
		new_memlimit = (uint32_t)new_memlimit * total_ram / 100;
	}

	if (set_compress) {
		memlimit_compress = new_memlimit;

#if SIZE_MAX == UINT32_MAX
		// FIXME?
		//
		// When running a 32-bit xz on a system with a lot of RAM and
		// using a percentage-based memory limit, the result can be
		// bigger than the 32-bit address space. Limiting the limit
		// below SIZE_MAX for compression (not decompression) makes
		// xz lower the compression settings (or number of threads)
		// to a level that *might* work. In practice it has worked
		// when using a 64-bit kernel that gives full 4 GiB address
		// space to 32-bit programs. In other situations this might
		// still be too high, like 32-bit kernels that may give much
		// less than 4 GiB to a single application.
		//
		// So this is an ugly hack but I will keep it here while
		// it does more good than bad.
		//
		// Use a value less than SIZE_MAX so that there's some room
		// for the xz program and so on. Don't use 4000 MiB because
		// it could look like someone mixed up base-2 and base-10.
#ifdef __mips__
		// For MIPS32, due to architectural peculiarities,
		// the limit is even lower.
		const uint64_t limit_max = UINT64_C(2000) << 20;
#else
		const uint64_t limit_max = UINT64_C(4020) << 20;
#endif

		// UINT64_MAX is a special case for the string "max" so
		// that has to be handled specially.
		if (memlimit_compress != UINT64_MAX
				&& memlimit_compress > limit_max)
			memlimit_compress = limit_max;
#endif
	}

	if (set_decompress)
		memlimit_decompress = new_memlimit;

	if (set_mtdec)
		memlimit_mtdec = new_memlimit;

	return;
}


extern uint64_t
hardware_memlimit_get(enum operation_mode mode)
{
	// 0 is a special value that indicates the default.
	// It disables the limit in single-threaded mode.
	//
	// NOTE: For multithreaded decompression, this is the hard limit
	// (memlimit_stop). hardware_memlimit_mtdec_get() gives the
	// soft limit (memlimit_threaded).
	const uint64_t memlimit = mode == MODE_COMPRESS
			? memlimit_compress : memlimit_decompress;
	return memlimit != 0 ? memlimit : UINT64_MAX;
}


extern uint64_t
hardware_memlimit_mtenc_get(void)
{
	return hardware_memlimit_mtenc_is_default()
			? memlimit_mt_default
			: hardware_memlimit_get(MODE_COMPRESS);
}


extern bool
hardware_memlimit_mtenc_is_default(void)
{
	return memlimit_compress == 0 && threads_are_automatic;
}


extern uint64_t
hardware_memlimit_mtdec_get(void)
{
	uint64_t m = memlimit_mtdec != 0
			? memlimit_mtdec
			: memlimit_mt_default;

	// Cap the value to memlimit_decompress if it has been specified.
	// This is nice for --info-memory. It wouldn't be needed for liblzma
	// since it does this anyway.
	if (memlimit_decompress != 0 && m > memlimit_decompress)
		m = memlimit_decompress;

	return m;
}


/// Helper for hardware_memlimit_show() to print one human-readable info line.
static void
memlimit_show(const char *str, size_t str_columns, uint64_t value)
{
	// Calculate the field width so that str will be padded to take
	// str_columns on the terminal.
	//
	// NOTE: If the string is invalid, this will be -1. Using -1 as
	// the field width is fine here so it's not handled specially.
	const int fw = tuklib_mbstr_fw(str, (int)(str_columns));

	// The memory usage limit is considered to be disabled if value
	// is 0 or UINT64_MAX. This might get a bit more complex once there
	// is threading support. See the comment in hardware_memlimit_get().
	if (value == 0 || value == UINT64_MAX)
		printf("  %-*s  %s\n", fw, str, _("Disabled"));
	else
		printf("  %-*s  %s MiB (%s B)\n", fw, str,
				uint64_to_str(round_up_to_mib(value), 0),
				uint64_to_str(value, 1));

	return;
}


extern void
hardware_memlimit_show(void)
{
	uint32_t cputhreads = 1;
#ifdef MYTHREAD_ENABLED
	cputhreads = lzma_cputhreads();
	if (cputhreads == 0)
		cputhreads = 1;
#endif

	if (opt_robot) {
		printf("%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
				"\t%" PRIu64 "\t%" PRIu32 "\n",
				total_ram,
				memlimit_compress,
				memlimit_decompress,
				hardware_memlimit_mtdec_get(),
				memlimit_mt_default,
				cputhreads);
	} else {
		const char *msgs[] = {
			_("Amount of physical memory (RAM):"),
			_("Number of processor threads:"),
			_("Compression:"),
			_("Decompression:"),
			_("Multi-threaded decompression:"),
			_("Default for -T0:"),
		};

		size_t width_max = 1;
		for (unsigned i = 0; i < ARRAY_SIZE(msgs); ++i) {
			size_t w = tuklib_mbstr_width(msgs[i], NULL);

			// When debugging, catch invalid strings with
			// an assertion. Otherwise fallback to 1 so
			// that the columns just won't be aligned.
			assert(w != (size_t)-1);
			if (w == (size_t)-1)
				w = 1;

			if (width_max < w)
				width_max = w;
		}

		puts(_("Hardware information:"));
		memlimit_show(msgs[0], width_max, total_ram);
		printf("  %-*s  %" PRIu32 "\n",
				tuklib_mbstr_fw(msgs[1], (int)(width_max)),
				msgs[1], cputhreads);

		putchar('\n');
		puts(_("Memory usage limits:"));
		memlimit_show(msgs[2], width_max, memlimit_compress);
		memlimit_show(msgs[3], width_max, memlimit_decompress);
		memlimit_show(msgs[4], width_max,
				hardware_memlimit_mtdec_get());
		memlimit_show(msgs[5], width_max, memlimit_mt_default);
	}

	tuklib_exit(E_SUCCESS, E_ERROR, message_verbosity_get() != V_SILENT);
}


extern void
hardware_init(void)
{
	// Get the amount of RAM. If we cannot determine it,
	// use the assumption defined by the configure script.
	total_ram = lzma_physmem();
	if (total_ram == 0)
		total_ram = (uint64_t)(ASSUME_RAM) * 1024 * 1024;

	// FIXME? There may be better methods to determine the default value.
	// One Linux-specific suggestion is to use MemAvailable from
	// /proc/meminfo as the starting point.
	memlimit_mt_default = total_ram / 4;

#if SIZE_MAX == UINT32_MAX
	// A too high value may cause 32-bit xz to run out of address space.
	// Use a conservative maximum value here. A few typical address space
	// sizes with Linux:
	//   - x86-64 with 32-bit xz: 4 GiB
	//   - x86: 3 GiB
	//   - MIPS32: 2 GiB
	const size_t mem_ceiling = 1400U << 20;
	if (memlimit_mt_default > mem_ceiling)
		memlimit_mt_default = mem_ceiling;
#endif

	return;
}
