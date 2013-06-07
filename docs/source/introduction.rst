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

''''''''''''
Requirements
''''''''''''
glymur works on Python 2.7 and 3.3.  Python 3.3 is strongly recommended.

OpenJPEG
========
OpenJPEG must be built as a shared library.  In addition, you
currently must compile OpenJPEG from the developmental source that
you can retrieve via subversion.  As of this time of writing, svn 
revision 2345 works.  In addition, you should also retrieve their test data, as
you will need it when running glymur's test suite.

Earlier versions of OpenJPEG through the 2.0 official release will **NOT**
work and are not supported.

Be sure to have the following ports/RPMs/debs installed.

    * gcc
    * gcc-c++
    * cmake
    
You should build OpenJPEG with testing turned on.  Consult the OpenJPEG 
documentation on how to do this.  If you use linux, make sure that you 
have the following development packages installed

    * zlib-devel
    * png-devel
    * libtiff-devel
    * lcms2-devel

OS
==

Mac OS X
--------
All the necessary packages are available to use glymur with Python 3.3 via
MacPorts.  A minimal set of ports includes

      * python33
      * py33-numpy
      * py33-distribute

To run all the testing, one of the following combinations of ports must
additionally be installed:

      * py33-scikit-image and either py33-Pillow or freeimage
      * py33-matplotlib and py33-Pillow

Linux
-----

Fedora 18
'''''''''
Fedora 18 ships with Python 3.3, so all the necessary RPMs are available to 
meet the minimal set of requirements.

      * python3 
      * python3-numpy
      * python3-setuptools
      * python3-matplotlib (for running tests)
      * python3-matplotlib-tk (or whichever matplotlib backend you prefer)

A few tests still will not run, however, unless one of the following
combinations of RPMs / Python packages is installed.

      * scikit-image and either Pillow or freeimage
      * matplotlib and Pillow

The 2nd route is probably the easiest, so go ahead and install Pillow
via pip since Pillow is not yet available in Fedora 18 default
repositories::

    $ yum install python3-devel       # pip needs this in order to compile Pillow
    $ yum install python3-pip
    $ pip-python3 install Pillow --user
    $ export PYTHONPATH=$HOME/.local/lib/python3.3/site-packages:$PYTHONPATH

Fedora 17
'''''''''
Fedora 17 ships with Python 3.2 and 2.7, so these steps detail working with 
2.7.  

Required RPMs include::

      * python
      * python-mock
      * python-pip
      * python-setuptools
      * numpy

In addition, you must install contextlib2 via pip.

A few tests still will not run, however, unless one of the following 
combinations of RPMs / Python packages is installed.

      * scikit-image and either Pillow or freeimage
      * matplotlib and Pillow

scikit-image is not available in the Fedora 17 default repositories, but 
it may be installed via pip::

    $ yum install Cython       # pip needs this in order to compile scikit-image
    $ yum install python-devel # pip needs this in order to compile scikit-image
    $ yum install freeimage    # scikit-image uses this as a backend
    $ yum install scipy        # needed by scikit-image
    $ pip-python install scikit-image --user
    $ pip-python install contextlib2 --user
    $ export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

Windows
-------
Not currently supported.

''''''''''''''''''''''''''''''''''''
Installation, Testing, Configuration
''''''''''''''''''''''''''''''''''''

From this point forward, python3 will be referred to as just "python".

Installation
============

You can install glymur via pip from the command line::

    $ pip install glymur

or manually retrieve the code from either of GitHub or PyPI

    * https://pypi.python.org/pypi/Glymur/ (stable releases)
    * http://github.com/quintusdias/glymur (bleeding edge)

and then unpack and install with::

    $ python setup.py install --prefix=/install/path

In addition to merely installing glymur, you should adjust your **$PATH**
environment variable in order to be able to use the *jp2dump* script from
the unix command line.

::

    $ export PYTHONPATH=/install/path/lib/python3.3/site-packages
    $ export PATH=/install/path/bin:$PATH


Configuration
=============
glymur uses ctypes (for the moment) to access the openjp2 library, and
because ctypes access libraries in a platform-dependent manner, it is 
recommended that you create a configuration file to help glymur properly find
the openjp2 library.  You may create the configuration file as follows::

    $ mkdir ~/.glymur
    $ cd ~/.glymur
    $ cat > glymurrc << EOF
    > [library]
    > openjp2: /opt/openjp2-svn/lib/libopenjp2.so
    > EOF

That assumes, of course, that you've installed OpenJPEG into /opt/openjp2-svn.


Testing
=======
In order to run all of the test suite, you will first need the OpenJPEG test
data that you previously retrieved.
Then you should set the **OPJ_DATA_ROOT** environment variable to
point to this directory, e.g. ::

    $ cd /somewhere/outside/the/glymur/unpacking/directory
    $ svn co http://openjpeg.googlecode.com/svn/data
    $ export OPJ_DATA_ROOT=`pwd`/data

The test suite may then be run with::

    $ cd /back/to/glymur/unpacking/directory
    $ python -m unittest discover

Quite a few tests are currently skipped.  These include tests whose
OpenJPEG counterparts are already failing, and others which do pass but
still produce heaps of output on stderr.  Rather than let this swamp
the signal (that most of the tests are actually passing), they've been
filtered out for now.  There are also more skipped tests on Python 2.7
than on Python 3.3.  The important part is whether or not any test
errors are reported at the end.
