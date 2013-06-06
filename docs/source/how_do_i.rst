------------
How do I...?
------------

Get the code?
=============
Go to http://github.com/quintusdias/glymur


Display metadata?
=================
There are two ways.  From the unix command line, the script *jp2dump* is
available. ::

    $ jp2dump /path/to/glymur/installation/data/nemo.jp2

From within Python, it is as simple as printing the Jp2k object, i.e. ::

    >>> import pkg_resources
    >>> from glymur import Jp2k
    >>> file = pkg_resources.resource_filename(glymur.__name__, "data/nemo.jp2")
    >>> j = Jp2k(file)
    >>> print(j)

This prints the metadata found in the JP2 boxes, but in the case of the
codestream box, only the main header is printed.  It is possible to print 
**only** the codestream information as well, i.e. ::

    >>> print(j.get_codestream())
