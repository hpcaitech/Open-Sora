# Installation

Please install the VMAF and FFMPEG package to base environment.

## VMAF

For calculating motion score, vmaf score, we use VMAF with FFMPEG, please follow the installation guides [VMAF](https://github.com/Netflix/vmaf/blob/master/libvmaf/README.md#install) and [here](https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md) and install the required FFMPEG software with VMAF support. Not that you need to export the path to VMAF before installing FFMPEG.
```
export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# change to the directory on your machine that contains libvmaf.so.3
```
