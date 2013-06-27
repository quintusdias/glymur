------------
How do I...?
------------


Read the lowest resolution thumbnail?
=====================================
Printing the Jp2k object should reveal the number of resolutions (look in the
COD segment section), but you can take a shortcut by supplying -1 as the reduce
level. ::

    >>> import glymur
    >>> file = glymur.data.nemo()
    >>> j = glymur.Jp2k(file)
    >>> thumbnail = j.read(reduce=-1)

Display metadata?
=================
There are two ways.  From the unix command line, the script *jp2dump* is
available. ::

    $ jp2dump /path/to/glymur/installation/data/nemo.jp2

From within Python, it is as simple as printing the Jp2k object, i.e. ::

    >>> from glymur import Jp2k
    >>> file = glymur.data.nemo()
    >>> j = Jp2k(file)
    >>> print(j)

This prints the metadata found in the JP2 boxes, but in the case of the
codestream box, only the main header is printed.  It is possible to print 
**only** the codestream information as well, i.e. ::

    >>> print(j.get_codestream())

Add XML Metadata?
=================
An existing raw codestream (or JP2 file) can be wrapped (re-wrapped) in a 
user-defined set of JP2 boxes.  To get just a minimal JP2 jacket on the 
codestream provided by `goodstuff.j2k`, you can use the **wrap** method with 
no box argument: ::

    >>> import glymur
    >>> jfile = glymur.data.goodstuff()
    >>> j2k = glymur.Jp2k(jfile)
    >>> jp2 = j2k.wrap("newfile.jp2")
    >>> print(jp2)
    File:  newfile.jp2
    JPEG 2000 Signature Box (jP  ) @ (0, 12)
        Signature:  0d0a870a
    File Type Box (ftyp) @ (12, 20)
        Brand:  jp2 
        Compatibility:  ['jp2 ']
    JP2 Header Box (jp2h) @ (32, 45)
        Image Header Box (ihdr) @ (40, 22)
            Size:  [800 480 3]
            Bitdepth:  8
            Signed:  False
            Compression:  wavelet
            Colorspace Unknown:  False
        Colour Specification Box (colr) @ (62, 15)
            Method:  enumerated colorspace
            Precedence:  0
            Colorspace:  sRGB
    Contiguous Codestream Box (jp2c) @ (77, 115228)
        Main header:
        .
        . (truncated)
        .

The raw codestream was wrapped in a JP2 jacket with four boxes in the outer
layer (the signature, file type, JP2 header, and contiguous codestream), with
two additional boxes (image header and color specification) contained in the
JP2 header superbox.

XML boxes are not in the minimal set of box requirements for the JP2 format, so
in order to add an XML box into the mix, we'll need to specify all of the
boxes.  If you already have a JP2 jacket in place, you can just reuse it,
though.  Take the following example content in an XML file `favorites.xml` : ::

    <?xml version="1.0"?>
    <favorite_things>
        <category>Light Ale</category>
    </favorite_things>

and add it after the JP2 header box, but before the codestream box ::

    >>> boxes = jp2.box  # The box attribute is the list of JP2 boxes
    >>> xmlbox = glymur.jp2box.XMLBox(file='favorites.xml')
    >>> boxes.insert(3, xmlbox)
    >>> jp2_xml = jp2.wrap("newfile_with_xml.jp2", boxes=boxes)
    >>> print(jp2_xml)
    File:  newfile_with_xml.jp2
    JPEG 2000 Signature Box (jP  ) @ (0, 12)
        Signature:  0d0a870a
    File Type Box (ftyp) @ (12, 20)
        Brand:  jp2 
        Compatibility:  ['jp2 ']
    JP2 Header Box (jp2h) @ (32, 45)
        Image Header Box (ihdr) @ (40, 22)
            Size:  [800 480 3]
            Bitdepth:  8
            Signed:  False
            Compression:  wavelet
            Colorspace Unknown:  False
        Colour Specification Box (colr) @ (62, 15)
            Method:  enumerated colorspace
            Precedence:  0
            Colorspace:  sRGB
    XML Box (xml ) @ (77, 76)
        <favorite_things>
          <category>Light Ale</category>
        </favorite_things>
        
    Contiguous Codestream Box (jp2c) @ (153, 115236)
        Main header:
        .
        . (truncated)
        .


Work with XMP UUIDs?
====================
The example JP2 file shipped with glymur has an XMP UUID. ::

    >>> from glymur import Jp2k
    >>> file = glymur.data.nemo()
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
