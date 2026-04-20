///////////////////////////////////////////////////////////////////////////////
//
/// \file       hardware_cputhreads.c
/// \brief      Get the number of CPU threads or cores
//
//  Author:     Lasse Collin
//
//  This file has been put into the public domain.
//  You can do whatever you want with this file.
//
///////////////////////////////////////////////////////////////////////////////

#include "common.h"

#include "tuklib_cpucores.h"


#ifdef HAVE_SYMBOL_VERSIONS_LINUX
// This is for compatibility with binaries linked against liblzma that
// has been patched with xz-5.2.2-compat-libs.patch from RHEL/CentOS 7.
LZMA_SYMVER_API("lzma_cputhreads@XZ_5.2.2",
	uint32_t, lzma_cputhreads_522)(void) lzma_nothrow
		__attribute__((__alias__("lzma_cputhreads_52")));

LZMA_SYMVER_API("lzma_cputhreads@@XZ_5.2",
	uint32_t, lzma_cputhreads_52)(void) lzma_nothrow;

#define lzma_cputhreads lzma_cputhreads_52
#endif
extern LZMA_API(uint32_t)
lzma_cputhreads(void)
{
	return tuklib_cpucores();
}
