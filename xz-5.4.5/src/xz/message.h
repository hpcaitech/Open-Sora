///////////////////////////////////////////////////////////////////////////////
//
/// \file       message.h
/// \brief      Printing messages to stderr
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

/// Verbosity levels
enum message_verbosity {
	V_SILENT,   ///< No messages
	V_ERROR,    ///< Only error messages
	V_WARNING,  ///< Errors and warnings
	V_VERBOSE,  ///< Errors, warnings, and verbose statistics
	V_DEBUG,    ///< Very verbose
};


/// \brief      Signals used for progress message handling
extern const int message_progress_sigs[];


/// \brief      Initializes the message functions
///
/// If an error occurs, this function doesn't return.
///
extern void message_init(void);


/// Increase verbosity level by one step unless it was at maximum.
extern void message_verbosity_increase(void);

/// Decrease verbosity level by one step unless it was at minimum.
extern void message_verbosity_decrease(void);

/// Get the current verbosity level.
extern enum message_verbosity message_verbosity_get(void);


/// \brief      Print a message if verbosity level is at least "verbosity"
///
/// This doesn't touch the exit status.
lzma_attribute((__format__(__printf__, 2, 3)))
extern void message(enum message_verbosity verbosity, const char *fmt, ...);


/// \brief      Prints a warning and possibly sets exit status
///
/// The message is printed only if verbosity level is at least V_WARNING.
/// The exit status is set to WARNING unless it was already at ERROR.
lzma_attribute((__format__(__printf__, 1, 2)))
extern void message_warning(const char *fmt, ...);


/// \brief      Prints an error message and sets exit status
///
/// The message is printed only if verbosity level is at least V_ERROR.
/// The exit status is set to ERROR.
lzma_attribute((__format__(__printf__, 1, 2)))
extern void message_error(const char *fmt, ...);


/// \brief      Prints an error message and exits with EXIT_ERROR
///
/// The message is printed only if verbosity level is at least V_ERROR.
tuklib_attr_noreturn
lzma_attribute((__format__(__printf__, 1, 2)))
extern void message_fatal(const char *fmt, ...);


/// Print an error message that an internal error occurred and exit with
/// EXIT_ERROR.
tuklib_attr_noreturn
extern void message_bug(void);


/// Print a message that establishing signal handlers failed, and exit with
/// exit status ERROR.
tuklib_attr_noreturn
extern void message_signal_handler(void);


/// Convert lzma_ret to a string.
extern const char *message_strm(lzma_ret code);


/// Display how much memory was needed and how much the limit was.
extern void message_mem_needed(enum message_verbosity v, uint64_t memusage);


/// Print the filter chain.
extern void message_filters_show(
		enum message_verbosity v, const lzma_filter *filters);


/// Print a message that user should try --help.
extern void message_try_help(void);


/// Prints the version number to stdout and exits with exit status SUCCESS.
tuklib_attr_noreturn
extern void message_version(void);


/// Print the help message.
tuklib_attr_noreturn
extern void message_help(bool long_help);


/// \brief      Set the total number of files to be processed
///
/// Standard input is counted as a file here. This is used when printing
/// the filename via message_filename().
extern void message_set_files(unsigned int files);


/// \brief      Set the name of the current file and possibly print it too
///
/// The name is printed immediately if --list was used or if --verbose
/// was used and stderr is a terminal. Even when the filename isn't printed,
/// it is stored so that it can be printed later if needed for progress
/// messages.
extern void message_filename(const char *src_name);


/// \brief      Start progress info handling
///
/// message_filename() must be called before this function to set
/// the filename.
///
/// This must be paired with a call to message_progress_end() before the
/// given *strm becomes invalid.
///
/// \param      strm      Pointer to lzma_stream used for the coding.
/// \param      in_size   Size of the input file, or zero if unknown.
///
extern void message_progress_start(lzma_stream *strm,
		bool is_passthru, uint64_t in_size);


/// Update the progress info if in verbose mode and enough time has passed
/// since the previous update. This can be called only when
/// message_progress_start() has already been used.
extern void message_progress_update(void);


/// \brief      Finishes the progress message if we were in verbose mode
///
/// \param      finished    True if the whole stream was successfully coded
///                         and output written to the output stream.
///
extern void message_progress_end(bool finished);
