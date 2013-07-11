----------------------------------
Detailed Installation Instructions
----------------------------------

''''''''''''''''''''''
Glymur Configuration
''''''''''''''''''''''

The default glymur installation process relies upon OpenJPEG version
1.5.1 being properly installed on your system.  This will, however,
only give you you basic read capabilities, so if you wish to take
advantage of more of glymur's features, you should compile OpenJPEG as
a shared library from the developmental source that you can retrieve
via subversion.  As of this time of writing, svn revision 2345 works.
You should also download the test data for the purpose of configuring
and running OpenJPEG's test suite, check their instructions for all this.
You should set the **OPJ_DATA_ROOT** environment variable for the purpose
of running Glymur's test suite. ::

    $ svn co http://openjpeg.googlecode.com/svn/data 
    $ export OPJ_DATA_ROOT=`pwd`/data

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

This assumes, of course, that you've installed OpenJPEG into
/opt/openjp2-svn on a linux system.  You may also substitute
**$XDG_CONFIG_HOME** for **$HOME/.config**.

You may also include a line for the version 1.5.1 library if you have it installed
in a non-standard place, i.e. ::

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
MacPorts.  A minimal set of ports includes

      * python33
      * py33-numpy
      * py33-distribute

To run all the testing, one of the following combinations of ports must
additionally be installed:

      * py33-scikit-image and either py33-Pillow or freeimage
      * py33-matplotlib and py33-Pillow

MacPorts supplies both OpenJPEG 1.5.0 and OpenJPEG 2.0.0.  As previously
mentioned, the 2.0.0 official release is not supported (although the 2.0+
development version via SVN *is* supported).

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
Fedora 17 ships with Python 3.2 and 2.7, but OpenJPEG is only at version 1.4,
so these steps detail working with Python 2.7 and the svn version of OpenJPEG.

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

scikit-image was not available in the Fedora 17 default repositories, but 
it was installable via pip::

    $ yum install Cython       # pip needs this in order to compile scikit-image
    $ yum install python-devel # pip needs this in order to compile scikit-image
    $ yum install freeimage    # scikit-image uses this as a backend
    $ yum install scipy        # needed by scikit-image
    $ pip-python install scikit-image --user
    $ pip-python install contextlib2 --user
    $ export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

Windows
-------
The only configuration I've tested is Python(xy), which uses Python 2.7.  
Python(xy) already comes with numpy, but you will have to install pip and then
contextlib2 and mock as well.  Both 1.5.1 and the svn development versions of
openjpeg work.


'''''''
Testing
'''''''

If you wish to run the tests (strongly suggested :-), you can either run them
from within python as follows ... ::

    >>> import glymur
    >>> glymur.runtests()

or from the unix command line. ::

    $ cd /to/where/you/unpacked/glymur
    $ python -m unittest discover

Quite a few tests are currently skipped.  These include tests whose
OpenJPEG counterparts are already failing, and others which do pass but
still produce heaps of output on stderr.  Rather than let this swamp
the signal (that most of the tests are actually passing), they've been
filtered out for now.  There are also more skipped tests on Python 2.7
than on Python 3.3.  The important part is whether or not any test
errors are reported at the end.
