----------------------------------------
Glymur: a Python interface for JPEG 2000
----------------------------------------

**Glymur** contains a Python interface to the OpenJPEG library
which allows linux and mac users to read and write JPEG 2000 files.  For more
information about OpenJPEG, please consult http://www.openjpeg.org.  Glymur
currently relies upon a development version of the OpenJPEG library, and so,
while useable, it is totally at the mercy of any upstream changes
made to the development version of OpenJPEG.

Glymur supports both reading and writing of JPEG 2000 images (part 1).  Writing
JPEG 2000 images is currently limited to images that can fit in memory,
however.

Of particular focus is retrieval of metadata.  Reading Exif UUIDs is supported,
as is reading XMP UUIDs as the XMP data packet is just XML.  There is
some very limited support for reading JPX metadata.  For instance,
**asoc** and **labl** boxes are recognized, so GMLJP2 metadata can
be retrieved from such JPX files.

Glymur works on Python 2.7 and 3.3.  Python 3.3 is strongly recommended.

OpenJPEG Installation
=====================
OpenJPEG must be built as a shared library.  In addition, you
currently must compile OpenJPEG from the developmental source that
you can retrieve via subversion.  As of this time of writing, svn 
revision 2345 works.  You should download the test data for the purpose
of configuring and running OpenJPEG's test suite, check their instructions for
all this.  You should set the **OPJ_DATA_ROOT** environment variable for the 
purpose of running Glymur's test suite. ::

    $ svn co http://openjpeg.googlecode.com/svn/data 
    $ export OPJ_DATA_ROOT=`pwd`/data

Earlier versions of OpenJPEG through the 2.0 official release will **NOT**
work and are not supported.

Glymur uses ctypes (for the moment) to access the openjp2 library, and
because ctypes access libraries in a platform-dependent manner, it is 
recommended that you create a configuration file to help Glymur properly find
the openjp2 library.  You may create the configuration file as follows::

    $ mkdir -p ~/.config/glymur
    $ cd ~/.config/glymur
    $ cat > glymurrc << EOF
    > [library]
    > openjp2: /opt/openjp2-svn/lib/libopenjp2.so
    > EOF

That assumes, of course, that you've installed OpenJPEG into /opt/openjp2-svn.
You may also replace **$HOME/.config** with **$XDG_CONFIG_HOME**.

Glymur Installation
===================
You can retrieve the source for Glymur from either of

    * https://pypi.python.org/pypi/Glymur/ (stable releases)
    * http://github.com/quintusdias/glymur (bleeding edge)

but you should now be able to get a functional installation of Glymur via
pip ::

    $ pip install glymur

This will install the **jp2dump** script, so you should adjust your **$PATH**
to take advantage of it.  For example, if you install with pip's
`--user` option on linux ::

    $ export PYTHONPATH=$HOME/.local/lib/python3.3/site-packages
    $ export PATH=$HOME/.local/bin:$PATH

You can run the tests from within python as follows::

    >>> import glymur
    >>> glymur.runtests()

Many tests are currently skipped; the important thing is whether or not any
tests fail.
