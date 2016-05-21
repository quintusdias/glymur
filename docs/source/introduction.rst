----------------------------------------
Glymur: a Python interface for JPEG 2000
----------------------------------------

**Glymur** is an interface to the OpenJPEG library
which allows one to read and write JPEG 2000 files from Python.  
Glymur supports both reading and writing of JPEG 2000 images, but writing
JPEG 2000 images is currently limited to images that can fit in memory.
**Glymur** can read images using OpenJPEG library versions as far back as 1.3,
but it is strongly recommended to use version 2.1.0, which is the most recently 
released version of OpenJPEG at this time.

In regards to metadata, most JP2 boxes are properly interpreted.
Certain optional JP2 boxes can also be written, including XML boxes and
XMP UUIDs.  There is incomplete support for reading JPX metadata.

Glymur works on Python versions 2.7, 3.3, 3.4, and 3.5.  If you have Python
2.6, you should use the 0.5 series of Glymur.

For more information about OpenJPEG, please consult http://www.openjpeg.org.

Glymur Installation
===================
Before installing Glymur, you should have OpenJPEG installed, so consult your
packager manager for this.  If you use Anaconda, please look consult 
https://anaconda.org/conda-forge/openjpeg.

The source for Glymur can be retrieved from either of

    * https://pypi.python.org/pypi/Glymur/ (stable releases)

and you should be able to install Glymur via pip ::

    $ pip install glymur

In addition to the package, this also gives you a command line script
**jp2dump** that can be used from the command line line to print JPEG 2000
metadata.
