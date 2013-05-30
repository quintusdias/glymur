************
How do I...?
************

Get the code?
=============
Go to http://github.com/quintusdias/glymur


Display metadata?
=================
There are two ways.  From the unix command line, the script **jp2dump** is
available. ::

    $ jp2dump /path/to/glymur/installation/data/nemo.jp2

From within Python, it is as simple as printing the Jp2k object, i.e. ::

    >>> import pkg_resources
    >>> from glymur import Jp2k
    >>> file = pkg_resources.resource_filename(glymur.__name__, "data/nemo.jp2")
    >>> j = Jp2k(file)
    >>> print(j)

The primary emphasis is on JP2 metadata, but it is possible to
display just raw codestream as well. This will display metadata present in the 
codestream's main header only. ::

    >>> print(j.get_codestream())
