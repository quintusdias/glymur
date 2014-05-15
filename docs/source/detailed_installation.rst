----------------------------------
Advanced Installation Instructions
----------------------------------

''''''''''''''''''''''
Glymur Configuration
''''''''''''''''''''''

The default glymur installation process relies upon OpenJPEG being
properly installed on your system as a shared library.  If you have OpenJPEG
installed through your system's package manager on linux or if you use MacPorts
on the mac, you are probably already set to go.  But if you have OpenJPEG 
installed into a non-standard place or if you use windows, then read on.

Glymur uses ctypes to access the openjp2/openjpeg libraries, and
because ctypes accesses libraries in a platform-dependent manner,
it is recommended that if you compile and install OpenJPEG into a
non-standard location,  you should then create a configuration file to
help Glymur properly find the openjpeg or openjp2 libraries (linux
users or macports users don't need to bother with this if you are
using OpenJPEG as provided by your package manager).  The configuration
format is the same as used by Python's configparser module,
i.e.  ::

    [library]
    openjp2: /opt/openjp2-svn/lib/libopenjp2.so

This assumes, of course, that you've installed OpenJPEG into
/opt/openjp2-svn on a linux system.  The location of the configuration file
can vary as well.  If you use either linux or mac, the path
to the configuration file would normally be ::

    $HOME/.config/glymur/glymurrc 

but if you have the **XDG_CONFIG_HOME** environment variable defined,
the path will be ::

    $XDG_CONFIG_HOME/glymur/glymurrc 

On windows, the path to the configuration file can be determined
by starting up Python and typing ::

    import os
    os.path.join(os.path.expanduser('~'), 'glymur', 'glymurrc')
        

You may also include a line for the version 1.x openjpeg library if you have it
installed in a non-standard place, i.e. ::

    [library]
    openjpeg: /not/the/usual/location/lib/libopenjpeg.so

'''''''
Testing
'''''''
It is not necessary, but you may wish to download OpenJPEG's test
data for the purpose of configuring and running OpenJPEG's test
suite.  Check their instructions on how to do that.  You can then
set the **OPJ_DATA_ROOT** environment variable for the purpose of
pointing Glymur to OpenJPEG's test suite. ::

    $ svn co http://openjpeg.googlecode.com/svn/data 
    $ export OPJ_DATA_ROOT=`pwd`/data

In order to run the tests, you can either run them from within
python as follows ... ::

    >>> import glymur
    >>> glymur.runtests()

or from the command line. ::

    $ cd /to/where/you/unpacked/glymur
    $ python -m unittest discover
