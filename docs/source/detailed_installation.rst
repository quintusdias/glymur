----------------------------------
Detailed Installation Instructions
----------------------------------

''''''''''''''''''''''
Glymur Configuration
''''''''''''''''''''''

The default glymur installation process relies upon OpenJPEG version
1.X being properly installed on your system.  This will, however, only
give you you basic read capabilities, so if you wish to take advantage
of more of glymur's features, you should compile OpenJPEG as a shared
library (named *openjp2* instead of *openjpeg*) from the developmental
source that you can retrieve via subversion.  As of this time of writing,
svn revision 2345 works.  You should also download the test data for
the purpose of configuring and running OpenJPEG's test suite, check
their instructions for all this.  You should set the **OPJ_DATA_ROOT**
environment variable for the purpose of running Glymur's test suite. ::

    $ svn co http://openjpeg.googlecode.com/svn/data 
    $ export OPJ_DATA_ROOT=`pwd`/data

Glymur uses ctypes (for the moment) to access the openjp2 library, and
because ctypes access libraries in a platform-dependent manner, it is 
recommended that you create a configuration file to help Glymur properly find
the openjp2 library.  The configuration format is the same as used by Python's
configparser module, i.e. ::

    [library]
    openjp2: /opt/openjp2-svn/lib/libopenjp2.so

This assumes, of course, that you've installed OpenJPEG into
/opt/openjp2-svn on a linux system.  You may also substitute
**$XDG_CONFIG_HOME** for **$HOME/.config**.

You may also include a line for the version 1.x openjpeg library if you have it
installed in a non-standard place, i.e. ::

    [library]
    openjp2: /opt/openjp2-svn/lib/libopenjp2.so
    openjpeg: /not/the/usual/location/lib/libopenjpeg.so

'''''''''''''''''''''''''''''''''''''''''''
Package Management Suggestions for Testing
'''''''''''''''''''''''''''''''''''''''''''

You only need to read this section if you want detailed 
platform-specific instructions on running as many tests as possible or wish to
use your system's package manager to install as many required 
packages/RPMs/ports/whatever without going through pip.


Mac OS X
--------
All the necessary packages are available to use glymur with Python 3.3 via
MacPorts.  You should install the following set of ports:

      * python33
      * py33-numpy
      * py33-distribute
      * py33-matplotlib (optional, for running certain tests)
      * py33-Pillow (optional, for running certain tests)

MacPorts supplies both OpenJPEG 1.5.0 and OpenJPEG 2.0.0.  As previously
mentioned, the 2.0.0 official release is not supported (although the 2.0+
development version via SVN *is* supported).

Linux
-----

Fedora 19
'''''''''
Fedora 18 ships with Python 3.3 and all the necessary RPMs are available to 
run the maximum number of tests.

      * python3 
      * python3-numpy
      * python3-setuptools
      * python3-matplotlib (for running tests)
      * python3-matplotlib-tk (or whichever matplotlib backend you prefer)
      * python3-pillow (for running tests)

Fedora 18
'''''''''
Fedora 18 ships with Python 3.3 and the following RPMs are available to 
meet the minimal set of requirements for running glymur.

      * python3 
      * python3-numpy
      * python3-setuptools

For running the maximal number of tests, you also need 

      * python3-matplotlib
      * python3-matplotlib-tk (or whichever matplotlib backend you prefer)

Pillow is also needed in order to run the maximum number of tests, so
go ahead and install Pillow via pip since Pillow is not available
in Fedora 18 default repositories::

    $ yum install python3-devel       # pip needs this in order to compile Pillow
    $ yum install python3-pip
    $ pip-python3 install Pillow --user
    $ export PYTHONPATH=$HOME/.local/lib/python3.3/site-packages:$PYTHONPATH

Fedora 17
'''''''''
Fedora 17 ships with Python 2.7 and OpenJPEG 1.4.  You should have the
following RPMs installed.

      * python
      * python-mock
      * python-pip
      * python-setuptools
      * numpy
      * matplotlib (optional)

In addition, you must install contextlib2 and Pillow via pip.

    $ yum install python-devel # pip needs this in order to compile Pillow
    $ pip-python install Pillow --user
    $ pip-python install contextlib2 --user
    $ export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

Windows
-------
The only configuration I've tested is Python(xy), which uses Python 2.7.  
Python(xy) already comes with numpy, but you will have to install pip and then
contextlib2 and mock as well.  Glymur seems to work with both 1.5.1 and the 
svn development versions of openjpeg.

Glymur has been tested **far less** extensively on Windows than on the other 
platforms.  


'''''''
Testing
'''''''

If you wish to run the tests (strongly recommended :-), you can either run them
from within python as follows ... ::

    >>> import glymur
    >>> glymur.runtests()

or from the command line. ::

    $ cd /to/where/you/unpacked/glymur
    $ python -m unittest discover

Quite a few tests are currently skipped.  These include tests whose
OpenJPEG counterparts are already failing, and others which do pass but
still produce heaps of output on stderr.  Rather than let this swamp
the signal (that most of the tests are actually passing), they've been
filtered out for now.  There are also more skipped tests on Python 2.7
than on Python 3.3.  The important part is whether or not any test
errors are reported at the end.
