///////////////////////////////////////////////////////////////////////////////
//
/// \file       hardware.h
/// \brief      Detection of available hardware resources
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

/// Initialize some hardware-specific variables, which are needed by other
/// hardware_* functions.
extern void hardware_init(void);


/// Set the maximum number of worker threads.
/// A special value of UINT32_MAX sets one thread in multi-threaded mode.
extern void hardware_threads_set(uint32_t threadlimit);

/// Get the maximum number of worker threads.
extern uint32_t hardware_threads_get(void);

/// Returns true if multithreaded mode should be used for .xz compression.
/// This can be true even if the number of threads is one.
extern bool hardware_threads_is_mt(void);


/// Set the memory usage limit. There are separate limits for compression,
/// decompression (also includes --list), and multithreaded decompression.
/// Any combination of these can be set with a single call to this function.
/// Zero indicates resetting the limit back to the defaults.
/// The limit can also be set as a percentage of installed RAM; the
/// percentage must be in the range [1, 100].
extern void hardware_memlimit_set(uint64_t new_memlimit,
		bool set_compress, bool set_decompress, bool set_mtdec,
		bool is_percentage);

/// Get the current memory usage limit for compression or decompression.
/// This is a hard limit that will not be exceeded. This is obeyed in
/// both single-threaded and multithreaded modes.
extern uint64_t hardware_memlimit_get(enum operation_mode mode);

/// This returns a system-specific default value if all of the following
/// conditions are true:
///
///   - An automatic number of threads was requested (--threads=0).
///
///   - --memlimit-compress wasn't used or it was reset to the default
///     value by setting it to 0.
///
/// Otherwise this is identical to hardware_memlimit_get(MODE_COMPRESS).
///
/// The idea is to keep automatic thread count reasonable so that too
/// high memory usage is avoided and, with 32-bit xz, running out of
/// address space is avoided.
extern uint64_t hardware_memlimit_mtenc_get(void);

/// Returns true if the value returned by hardware_memlimit_mtenc_get() is
/// a system-specific default value. coder.c uses this to ignore the default
/// memlimit in case it's too small even for a single thread in multithreaded
/// mode. This way the default limit will never make xz fail or affect the
/// compressed output; it will only make xz reduce the number of threads.
extern bool hardware_memlimit_mtenc_is_default(void);

/// Get the current memory usage limit for multithreaded decompression.
/// This is only used to reduce the number of threads. This limit can be
/// exceeded if the number of threads are reduce to one. Then the value
/// from hardware_memlimit_get() will be honored like in single-threaded mode.
extern uint64_t hardware_memlimit_mtdec_get(void);

/// Display the amount of RAM and memory usage limits and exit.
tuklib_attr_noreturn
extern void hardware_memlimit_show(void);
