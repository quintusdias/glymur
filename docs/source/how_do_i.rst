------------
How do I...?
------------


... read the lowest resolution thumbnail?
=========================================
Printing the Jp2k object should reveal the number of resolutions (look in the
COD segment section), but you can take a shortcut by supplying -1 as the
resolution level. ::

    >>> import glymur
    >>> file = glymur.data.nemo()
    >>> j = glymur.Jp2k(file)
    >>> thumbnail = j.read(rlevel=-1)

... display metadata?
=====================
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

... add XML metadata?
=====================
You can append any number of XML boxes to a JP2 file (not to a raw codestream).
Consider the following XML file `data.xml` : ::

    <?xml version="1.0"?>
    <info>
        <locality>
            <city>Boston</city>
            <snowfall>24.9 inches</snowfall>
        </locality>
        <locality>
            <city>Portland</city>
            <snowfall>31.9 inches</snowfall>
        </locality>
        <locality>
            <city>New York City</city>
            <snowfall>11.4 inches</snowfall>
        </locality>
    </info>

The **append** method can add an XML box as shown below::

    >>> import shutil
    >>> import glymur
    >>> shutil.copyfile(glymur.data.nemo(), 'myfile.jp2')
    >>> from xml.etree import cElementTree as ET
    >>> jp2 = glymur.Jp2k('myfile.jp2')
    >>> xmlbox = glymur.jp2box.XMLBox(filename='data.xml')
    >>> jp2.append(xmlbox)
    >>> print(jp2)

... add metadata in a more general fashion?
===========================================
An existing raw codestream or JP2 file can be wrapped (re-wrapped in the case
of JP2) in a user-defined set of JP2 boxes.  To get just a minimal
JP2 jacket on the codestream provided by `goodstuff.j2k` (a file
consisting of a raw codestream), you can use the **wrap** method
with no box argument: ::

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

XML boxes are not in the minimal set of box requirements for the
JP2 format, so in order to add an XML box into the mix before the
codestream box, we'll need to re-specify all of the boxes.  If you
already have a JP2 jacket in place, you can just reuse that, though.
Take the following example content in an XML file `favorites.xml`
: ::

    <?xml version="1.0"?>
    <favorite_things>
        <category>Light Ale</category>
    </favorite_things>

In order to add the XML after the JP2 header box, but before the codestream box, 
the following will work. ::

    >>> boxes = jp2.box  # The box attribute is the list of JP2 boxes
    >>> xmlbox = glymur.jp2box.XMLBox(filename='favorites.xml')
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

As to the question of which method you should use, **append** or **wrap**,
to add metadata, you should keep in mind that **wrap** produces a new JP2 file,
while **append** modifies an existing file and is currently limited to XML
boxes.

... create an image with an alpha layer?
========================================

OpenJPEG can create JP2 files with more than 3 components (requires version
2.1), but by default any extra components are not described as such by the JP2
boxes created by OpenJPEG.  In order to do so, we need to rewrap such
an image in a set of boxes that includes a channel definition box.

This example is based on SciPy example code found at 
http://scipy-lectures.github.io/advanced/image_processing/#basic-manipulations . 
Instead of a circular mask we'll make it an ellipse since the source
image isn't square. ::

    >>> import numpy as np
    >>> import glymur
    >>> from glymur import Jp2k
    >>> rgb = Jp2k(glymur.data.goodstuff()).read()
    >>> lx, ly = rgb.shape[0:2]
    >>> X, Y = np.ogrid[0:lx, 0:ly]
    >>> mask = ly**2*(X - lx / 2) ** 2 + lx**2*(Y - ly / 2) ** 2 > (lx * ly / 2)**2
    >>> alpha = 255 * np.ones((lx, ly, 1), dtype=np.uint8)
    >>> alpha[mask] = 0
    >>> rgba = np.concatenate((rgb, alpha), axis=2)
    >>> jp2 = Jp2k('tmp.jp2', 'wb')
    >>> jp2.write(rgba)

Next we need to specify what types of channels we have.
The first three channels are color channels, but we identify the fourth as
an alpha channel::

    >>> from glymur.core import COLOR, OPACITY
    >>> ctype = [COLOR, COLOR, COLOR, OPACITY]

And finally we have to specify just exactly how each channel is to be
interpreted.  The color channels are straightforward, they correspond to R-G-B,
but the alpha (or opacity) channel in this case is to be applied against the 
entire image (it is possible to apply an alpha channel to a single color 
channel, but we aren't doing that). ::

    >>> from glymur.core import RED, GREEN, BLUE, WHOLE_IMAGE
    >>> asoc = [RED, GREEN, BLUE, WHOLE_IMAGE]
    >>> cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=ctype, association=asoc)
    >>> print(cdef)
    Channel Definition Box (cdef) @ (0, 0)
        Channel 0 (color) ==> (1)
        Channel 1 (color) ==> (2)
        Channel 2 (color) ==> (3)
        Channel 3 (opacity) ==> (whole image)

It's easiest to take the existing jp2 jacket and just add the channel
definition box in the appropriate spot.  The channel definition box **must**
go into the jp2 header box, and then we can rewrap the image. ::

    >>> boxes = jp2.box  # The box attribute is the list of JP2 boxes
    >>> boxes[2].box.append(cdef)
    >>> jp2_rgba = jp2.wrap("goodstuff_rgba.jp2", boxes=boxes)

Here's how the Preview application on the mac shows the RGBA image.

.. image:: goodstuff_alpha.png

    
work with XMP UUIDs?
====================
The example JP2 file shipped with glymur has an XMP UUID. ::

    >>> import glymur
    >>> j = glymur.Jp2k(glymur.data.nemo())
    >>> print(j.box[4])
    UUID Box (uuid) @ (715, 2412)
        UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
        UUID Data:  
        <ns0:xmpmeta xmlns:ns0="adobe:ns:meta/" xmlns:ns2="http://ns.adobe.com/xap/1.0/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" ns0:xmptk="XMP Core 4.4.0-Exiv2">
          <rdf:RDF>
            <rdf:Description ns2:CreatorTool="glymur" rdf:about="" />
          </rdf:RDF>
        </ns0:xmpmeta>

Since the UUID data in this case is returned as an ElementTree instance, one can
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
