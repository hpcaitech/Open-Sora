///////////////////////////////////////////////////////////////////////////////
//
/// \file       test_hardware.c
/// \brief      Tests src/liblzma/api/lzma/hardware.h API functions
///
/// Since the output values of these functions are hardware dependent, these
/// tests are trivial. They are simply used to detect errors and machines
/// that these function are not supported on.
//
//  Author:     Jia Tan
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "tests.h"
#include "mythread.h"


static void
test_lzma_physmem(void)
{
	// NOTE: Use _skip instead of _fail because 0 can also mean that we
	// don't know how to get this information on this operating system.
	if (lzma_physmem() == 0)
		assert_skip("Could not determine amount of physical memory");
}


static void
test_lzma_cputhreads(void)
{
#ifndef MYTHREAD_ENABLED
	assert_skip("Threading support disabled");
#else
	if (lzma_cputhreads() == 0)
		assert_skip("Could not determine cpu core count");
#endif
}


extern int
main(int argc, char **argv)
{
	tuktest_start(argc, argv);
	tuktest_run(test_lzma_physmem);
	tuktest_run(test_lzma_cputhreads);
	return tuktest_end();
}
