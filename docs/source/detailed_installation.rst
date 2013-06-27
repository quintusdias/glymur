-----------------------------------------------------
Detailed Instructions for Package Management, Testing
-----------------------------------------------------

You only need to read this page if you want detailed 
platform-specific instructions on running as many tests as possible or wish to
use your system's package manager to install as many required 
packages/RPMs/ports/whatever without going through pip.  Otherwise go on to
the next page.

''''''''
Platform
''''''''

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

Raspbian
''''''''
Yeah, this was the first thing I tried after getting my new Raspberry Pi hooked
up (couldn't help myself :-)  Raspbian ships with Python 3.2 and 2.7, so these steps detail working with 2.7.

Additional required OS packages include::

    * python-pip
    * python-pkg-resources
    * python-mock

You must install contextlib2 via pip, and then you can run at least
a minimal number of tests.  To attempt to run more of the tests,
install the following debs::

    * python-dev
    * python-matplotlib

and then install Pillow via pip.  The tests take about 30 minutes to run, with
one unexpected failure as of the time of writing.

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
Not currently supported.

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
