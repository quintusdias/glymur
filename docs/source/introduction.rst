########################################
Glymur: a Python interface for JPEG 2000
########################################

**Glymur** is an interface to the OpenJPEG library
which allows one to read and write JPEG 2000 files from Python.  
Glymur supports both reading and writing of JPEG 2000 images.  A historical
limitation of glymur was that it could not write images that did not fit
into memory, but that limitation has been removed.

In regards to metadata, most JP2 boxes are properly interpreted.
Certain optional JP2 boxes can also be written, including XML boxes and
XMP UUIDs.  There is incomplete support for reading JPX metadata.

The current version of glymur works on Python versions 3.7, 3.8, 3.9,
and 3.10.

For more information about OpenJPEG, please consult http://www.openjpeg.org.

*******************
Glymur Installation
*******************
The easiest way to install Glymur is via Anaconda using conda-forge ::

    $ conda create -n testglymur -c conda-forge python glymur
    $ conda activate testglymur

