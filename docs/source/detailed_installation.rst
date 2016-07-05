----------------------------------
Advanced Installation Instructions
----------------------------------

''''''''''''''''''''''
Glymur Configuration
''''''''''''''''''''''
If you installed OpenJPEG via conda, you don't have to do any
configuration, as glymur can find the OpenJPEG library within the
Anaconda directory structure.

Otherwise, the default glymur installation process relies upon OpenJPEG being
properly installed on your system as a shared library. If you have
OpenJPEG installed through your system’s package manager on linux, Cygwin,
or if you use MacPorts on the mac, you are probably already set to
go. But if you have OpenJPEG installed into a non-standard place
or if you use windows, then read on.

Glymur uses ctypes to access the openjp2/openjpeg libraries, and
because ctypes accesses libraries in a platform-dependent manner,
it is recommended that **if** you compile and install OpenJPEG into a
non-standard location, you should then create a configuration file
to help Glymur properly find the openjpeg or openjp2 libraries The
configuration format is the same as used by Python’s configparser
module, i.e. ::

    [library]
    openjp2: /somewhere/lib/libopenjp2.so

This assumes, of course, that you've installed OpenJPEG into
/somewhere/lib on a linux system.  The location of the configuration file
can vary as well.  If you use either linux or mac, the path
to the configuration file would normally be ::

    $HOME/.config/glymur/glymurrc 

but if you have the **XDG_CONFIG_HOME** environment variable defined,
the path will be ::

    $XDG_CONFIG_HOME/glymur/glymurrc 

On windows, the path to the configuration file can be determined by starting
up Python and typing ::

    import os
    os.path.join(os.path.expanduser('~'), 'glymur', 'glymurrc')

You may also include a line for the version 1.x openjpeg library if you have it
installed in a non-standard place, i.e. ::

    [library]
    openjpeg: /somewhere/lib/libopenjpeg.so

Once again, you should not have to bother with a configuration file if you use
mac, linux, or Cygwin, and OpenJPEG is provided by your package manager.
