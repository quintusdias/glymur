----------------------------------
Advanced Installation Instructions
----------------------------------
Most users won't need to read this!  You've been warned...

''''''''''''''''''''''
Glymur Configuration
''''''''''''''''''''''

The default glymur installation process relies upon OpenJPEG
being properly installed on your system.  If you have version 1.5 you can
both read and write JPEG 2000 files, but you may wish to install version 2.0 
or the 2.0+ version from OpenJPEG's development trunk for better performance.
If you do that, you should compile it as a shared library (named *openjp2*
instead of *openjpeg*) from the developmental source that you can retrieve
via subversion.  As of this time of writing, svn revision r2651 works.
You should also download the test data for the purpose of configuring
and running OpenJPEG's test suite, check their instructions for all this.
You should set the **OPJ_DATA_ROOT** environment variable for the purpose
of running Glymur's test suite. ::

    $ svn co http://openjpeg.googlecode.com/svn/data 
    $ export OPJ_DATA_ROOT=`pwd`/data

Glymur uses ctypes to access the openjp2/openjpeg libraries,
and because ctypes accesses libraries in a platform-dependent manner, it is 
recommended that you create a configuration file to help Glymur properly find
the openjpeg or openjp2 libraries (linux users don't need to bother with this 
if you are using OpenJPEG as provided by your package manager).  The 
configuration format is the same as used by Python's configparser module, 
i.e.  ::

    [library]
    openjp2: /opt/openjp2-svn/lib/libopenjp2.so

This assumes, of course, that you've installed OpenJPEG into
/opt/openjp2-svn on a linux system.  The location of the configuration file
can vary as well (of course).  If you use either linux or mac, the path
to the configuration file would normally be ::

    $HOME/.config/glymur/glymurrc 

but if you have the **XDG_CONFIG_HOME** environment variable defined,
the path will be ::

    $XDG_CONFIG_HOME/glymur/glymurrc 

You may also include a line for the version 1.x openjpeg library if you have it
installed in a non-standard place, i.e. ::

    [library]
    openjpeg: /not/the/usual/location/lib/libopenjpeg.so

''''''''''''''''''''''''''''''
Package Management Suggestions
''''''''''''''''''''''''''''''

You only need to read this section if you want detailed 
platform-specific instructions on running as many tests as possible or wish to
use your system's package manager to install as many required 
packages/RPMs/ports/whatever without going through pip.


Mac OS X
--------
All the necessary packages are available to use glymur with MacPorts.
For python 3.3, you should install the following set of ports:

      * python33
      * py33-numpy
      * py33-lxml
      * py33-distribute
      * py33-Pillow (optional, for running certain tests)

MacPorts supplies both OpenJPEG 1.5.0 and OpenJPEG 2.0.0.

Linux
-----
For the most part, you only need python and numpy to run glymur, so on
just about all distributions you are already set to go (and you don't
need to mess around with a configuration file, as the openjpeg shared
libraries are found in the usual places thanks to your package manager).
In order to run as many tests as possible, however, the following Python
packages may also need to be installed.  Consult your package manager
documentation or use pip.

      * setuptools
      * python-lxml
      * matplotlib
      * pillow
      * contextlib2 (2.7 only)
      * mock (2.7 only)

Glymur 0.6 been tested on the following linux platforms without any unexpected
difficulties:
 
      * OpenSUSE 13.1
      * Fedora 19
      * Raspian
      * Travis CI (currently Ubuntu 12.04?)

Windows
-------
The 0.6.x series of Glymur is untested on windows and I make no promises here.
I suggest that windows users check the 0.5.x series.


'''''''
Testing
'''''''

There are two environment variables you may wish to set before running the
tests.  

    * **OPJ_DATA_ROOT** - points to directory for OpenJPEG test data (see above)
    * **FORMAT_CORPUS_DATA_ROOT** - points to directory for format-corpus repository  (see https://github.com/openplanets/format-corpus if you wish, but you really don't need to bother with this)

Setting these two environment variables is not required, as any tests using 
either of them will be skipped.

In order to run the tests, you can either run them from within
python as follows ... ::

    >>> import glymur
    >>> glymur.runtests()

or from the command line. ::

    $ cd /to/where/you/unpacked/glymur
    $ python -m unittest discover

Quite a few tests are currently skipped.  These include tests whose
OpenJPEG counterparts are already failing, and others which do pass but
still produce heaps of output on stderr.  Rather than let this swamp
the signal (that most of those tests are actually passing), they've been
filtered out for now.  There are also more skipped tests on Python 2.7
than on Python3.  The important part is whether or not any test
errors are reported at the end.
