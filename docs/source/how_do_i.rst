************
How do I...?
************

Display metadata?
=================
There are two ways.  From the unix command line, the script **jp2dump** is
available.::

    $ jp2dump /path/to/glymur/installation/data/nemo.jp2

From within Python, it is as simple as printing the Jp2k object, i.e.::

    >>> import pkg_resources
    >>> from glymur import Jp2k
    >>> file = pkg_resources.resource_filename(glymur.__name__, "data/nemo.jp2")
    >>> j = Jp2k(file)
    >>> print(j)

The primary emphasis is on JP2 metadata, but it is possible to
display raw codestream as well::

    >>> print(j.get_codestream())

