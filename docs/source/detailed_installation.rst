----------------------------------
Advanced Installation Instructions
----------------------------------

''''''''''''''''''''''
Glymur Configuration
''''''''''''''''''''''

The default glymur installation process relies upon OpenJPEG
being properly installed on your system.  If you have version 1.5 you can
both read and write JPEG 2000 files, but version 2.1 is recommended.
If you compile OpenJPEG yourself, please compile it as a shared library.
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
