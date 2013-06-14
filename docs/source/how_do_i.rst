------------
How do I...?
------------


Read the lowest resolution thumbnail?
=====================================
Printing the Jp2k object should reveal the number of resolutions (look in the
COD segment section), but you can take a shortcut by supplying -1 as the reduce
level. ::

    >>> import pkg_resources
    >>> import glymur
    >>> file = pkg_resources.resource_filename(glymur.__name__, "data/nemo.jp2")
    >>> j = glymur.Jp2k(file)
    >>> thumbnail = j.read(reduce=-1)

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

Work with XMP UUIDs?
====================
The example JP2 file shipped with glymur has an XMP UUID. ::

    >>> import pkg_resources
    >>> from glymur import Jp2k
    >>> file = pkg_resources.resource_filename(glymur.__name__, "data/nemo.jp2")
    >>> j = Jp2k(file)
    >>> print(j.box[4])
    UUID Box (uuid) @ (715, 2412)
        UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
        UUID Data:  
        <ns0:xmpmeta xmlns:ns0="adobe:ns:meta/" xmlns:ns2="http://ns.adobe.com/xap/1.0/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" ns0:xmptk="XMP Core 4.4.0-Exiv2">
          <rdf:RDF>
            <rdf:Description ns2:CreatorTool="glymur" rdf:about="" />
          </rdf:RDF>
        </ns0:xmpmeta>

Since the UUID data in this case is returned as an ElementTree Element, one can
use ElementTree to access the data.  For example, to extract the 
**CreatorTool** attribute value, the following would work::

    >>> xmp = j.box[4].data
    >>> ns0 = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    >>> ns1 = '{http://ns.adobe.com/xap/1.0/}'
    >>> name = '{0}RDF/{0}Description'.format(ns0)
    >>> elt = xmp.find(name)
    >>> elt
    <Element '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description' at 0xb4baa93c>
    >>> elt.attrib['{0}CreatorTool'.format(ns1)]
    'glymur'
