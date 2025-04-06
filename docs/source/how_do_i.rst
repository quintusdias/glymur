########
Examples
########

***************
Basic Image I/O
***************

How do I...?
============

... read images?
----------------

Jp2k implements slicing via the :py:meth:`__getitem__` method and
hooks it into the multiple resolution property of JPEG 2000 imagery.
This allows you to retrieve multiresolution imagery via
array-style slicing, i.e. strides.  For example, here's how
to retrieve a full resolution, first lower-resolution image, and
second lower-resolution image.  A stride will always be a power of two. ::

    >>> import glymur
    >>> j2kfile = glymur.data.goodstuff() # just a path to a raw JPEG 2000 codestream
    >>> j2k = glymur.Jp2k(j2kfile)
    >>> fullres = j2k[:]
    >>> fullres.shape
    (800, 480, 3)
    >>> thumbnail = j2k[::2, ::2]
    >>> thumbnail.shape
    (400, 240, 3)
    >>> thumbnail2 = j2k[::4, ::4]
    >>> thumbnail2.shape
    (200, 120, 3)


... read really large images
----------------------------

JPEG 2000 images can be much larger than what can fit into your
computer's memory.  While you can use strides that align with the
JPEG 2000 decomposition levels to retrieve lower resolution images,
retrieving the lowest resolution image would seem to require that
you know just how many decomposition levels are available.  While
you can get that information from the COD segment in the codestream,
glymur provides you with a shortcut.  Normally the stride must be
a power of 2, but you can provide -1 instead to get the smallest
thumbnail.::

    >>> import glymur
    >>> j2kfile = glymur.data.goodstuff() # just a path to a JPEG 2000 file
    >>> j2k = glymur.Jp2k(j2kfile)
    >>> j2k.shape
    (800, 480, 3)
    >>> j2k.codestream.segment[2].num_res
    5
    >>> j2k[::32, ::32].shape
    (25, 15, 3)
    >>> thumbnail = j2k[::-1, ::-1] # last thumbnail was the 5th, 2 ** 5 = 32
    >>> thumbnail.shape
    (25, 15, 3)

... read an image layer?
------------------------

JPEG2000 has layers which allow you to specify images with different
levels of quality.  Different layers may be specified by utilizing 
the layer property.  The default layer value is 0, which specifies the
first layer. ::

    >>> import glymur
    >>> file = glymur.data.jpxfile() # just a path to a JPEG2000 file
    >>> jpx = glymur.Jp2k(file)
    >>> d0 = jpx[:] # first layer
    >>> jpx.layer = 3
    >>> d3 = jpx[:] # third layer

... write images?
-----------------

The easiest way is just to assign the entire image, similar to what might
be done with NumPy. ::
    
    >>> import glymur, skimage.data
    >>> jp2 = glymur.Jp2k('astronaut.jp2')
    >>> jp2[:] = skimage.data.astronaut()

******************
Advanced Image I/O
******************

How do I...?
============

... make use of OpenJPEG's thread support to read images?
---------------------------------------------------------

If you have glymur 0.8.13 or higher
and OpenJPEG 2.2.0 or higher,
you can make use of OpenJPEG's thread support to speed-up read operations.
If you have really big images and a large number of cores at your disposal,
you really should look into this. ::

    >>> import glymur, time
    >>> jp2file = glymur.data.nemo()
    >>> jp2 = glymur.Jp2k(jp2file)
    >>> t0 = time.time(); data = jp2[:]; t1 = time.time()
    >>> t1 - t0  # doctest: +SKIP
    0.9024193286895752
    >>> glymur.set_option('lib.num_threads', 2)
    >>> t0 = time.time(); data = jp2[:]; t1 = time.time()
    >>> t1 - t0  # doctest: +SKIP
    0.4060473537445068
    >>> glymur.reset_option('all')


... efficiently read just one band of a big image?
--------------------------------------------------

For really large images, before v0.9.4 you had to read in all bands of an
image, even if you were only interested in just one of those bands.  With
v0.9.4 or higher, you can make use of the :py:meth:`decoded_components`
property, which will inform the openjpeg library to just decode the
specified component(s), which can significantly speed up read operations
on large images.  Be aware, however, that the openjpeg library will not
employ the MCT when decoding these components.

You can set the property to None to restore the behavior of decoding all
bands.

    >>> jp2file = glymur.data.nemo()
    >>> jp2 = glymur.Jp2k(jp2file)
    >>> data = jp2[:]
    >>> data.shape
    (1456, 2592, 3)
    >>> jp2.decoded_components = 1
    >>> data = jp2[:]
    >>> data.shape
    (1456, 2592)
    >>> jp2.decoded_components = [0, 2]
    >>> data = jp2[:]
    >>> data.shape
    (1456, 2592, 2)
    >>> jp2.decoded_components = None
    >>> data = jp2[:]
    >>> data.shape
    (1456, 2592, 3)

... write images using multithreaded encoding?
----------------------------------------------

If you have glymur 0.9.3 or higher
and OpenJPEG 2.4.0 or higher,
you can make use of OpenJPEG's thread support to speed-up read operations.
With a puny 2015 macbook, just two cores, and a 5824x10368x3 image, we get::

    >>> import glymur, time, numpy as np
    >>> data = glymur.Jp2k(glymur.data.nemo())[:]
    >>> data = np.tile(data, (4, 4, 1))
    >>> t0 = time.time()
    >>> j = glymur.Jp2k('1thread.jp2', data=data)
    >>> t1 = time.time()
    >>> print(f'1 thread:  {(t1 - t0):.3} seconds')  # doctest: +SKIP
    12.0 seconds
    >>> t0 = time.time()
    >>> glymur.set_option('lib.num_threads', 2)
    >>> j = glymur.Jp2k('2threads.jp2', data=data)
    >>> t1 = time.time()
    >>> print(f'2 threads:  {(t1 - t0):.3} seconds')  # doctest: +SKIP
    7.24 seconds


... write images that cannot fit into memory?
---------------------------------------------

If you have glymur 0.9.4 or higher, you can write out an image tile-by-tile.
In this example, we take a 512x512x3 image and tile it into a 2x2 grid,
resulting in a 1024x1024x3 image, but we could have just as easily tiled it
20x20 or 100x100.  Consider setting py::meth::`verbose` to
True to get detailed feedback from the OpenJPEG library as to which tile is
currently being written. ::

    >>> import glymur, skimage.data
    >>> from glymur import Jp2k
    >>> img = skimage.data.astronaut()
    >>> print(img.shape)
    (512, 512, 3)
    >>> shape = img.shape[0] * 20, img.shape[1] * 20, 3
    >>> tilesize = (img.shape[0], img.shape[1])
    >>> j = Jp2k('4astronauts.jp2', shape=shape, tilesize=tilesize)
    >>> for tw in j.get_tilewriters():
    ...     tw[:] = img
    >>> j = Jp2k('4astronauts.jp2')
    >>> print(j.shape)
    (10240, 10240, 3)

Note that the tiles are written out left-to-right, tile-row-by-tile-row.  You must
have image data ready to feed each tile writer, you cannot skip a tile.

... force the generation of PLT markers?
----------------------------------------

With glymur 0.9.5 or higher and openjpeg 2.4.0 or higher, you can instruct the
encoder to generate PLT markers by using the plt keyword. ::

    >>> import glymur, skimage.data
    >>> if glymur.version.openjpeg_version >= '2.4.0':
    ...     jp2 = glymur.Jp2k('plt.jp2', plt=True)
    ...     jp2[:] = skimage.data.astronaut()
    ...     c = jp2.get_codestream(header_only=False)
    ...     print(c.segment[6])  # doctest: +SKIP
    PLT marker segment @ (222, 45)
        Index:  0
        Iplt:  [271, 201, 208, 749, 551, 548, 2569, 1852, 1814, 8300, 6370, 6061, 26987, 23437, 21431, 88511, 86763, 77253]

... write images with different compression ratios for different layers?
------------------------------------------------------------------------

Different compression factors may be specified with the cratios parameter ::

    >>> import glymur, skimage.data
    >>> data = skimage.data.camera()
    >>> # quality layer 1: compress 20x
    >>> # quality layer 2: compress 10x
    >>> # quality layer 3: compress lossless
    >>> jp2 = glymur.Jp2k('compress.jp2', data=data, cratios=[20, 10, 1])
    >>> # read the lossless layer
    >>> jp2.layer = 2
    >>> data = jp2[:]

... write images with different PSNR (or "quality") for different layers?
-------------------------------------------------------------------------

Different PSNR values may be specified with the psnr parameter.  Please read
https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
for a basic understanding of PSNR.  

Values must be increasing, but the last value may be 0 to indicate
the layer is lossless.  However, the OpenJPEG library will reorder
the layers to make the first layer lossless, not the last.

We suppress a harmless warning from scikit-image below. ::

    >>> import glymur, skimage.data, skimage.metrics, warnings
    >>> warnings.simplefilter('ignore')
    >>> truth = skimage.data.camera()
    >>> jp2 = glymur.Jp2k('psnr.jp2', data=truth, psnr=[30, 40, 50, 0])
    >>> psnr = []
    >>> for layer in range(4):
    ...     jp2.layer = layer
    ...     psnr.append(skimage.metrics.peak_signal_noise_ratio(truth, jp2[:]))
    >>> print(psnr)                # doctest: +SKIP
    [inf, 29.90221522329731, 39.71824592284344, 48.381047443043634]


... convert JPEG or TIFF images to JPEG 2000?
---------------------------------------------

Many JPEGs and TIFFs can be converted to tiled JPEG 2000 files using glymur.
Command line utilities **jpeg2jp2** and **tiff2jp2** are provided for this task.
TIFF conversion is described here, but JPEG conversion is similar.

In most cases, you should provide your own choice of a JPEG 2000 tile
size.  Not providing a tile size will cause glymur to try to covert
the TIFF into a single-tile JPEG 2000 file.  If your TIFF is large,
you may not have enough memory to write such a single-tile file. ::

    $ wget http://photojournal.jpl.nasa.gov/tiff/PIA17145.tif
    $ tiff2jp2 --tilesize 256 256 PIA17145.tif PIA17145.jp2

If your TIFF is really big but has an unfortunate choice for the
RowsPerStrip tag (like the seemingly ubiquitous value of 3, which was
reasonable only in prehistoric times) ... well that's going to be very
inefficient no matter how you tile the JPEG 2000 file.

The TIFF metadata is stored in UUID boxes appended to the end of the
JPEG 2000 file.

... create an image with an alpha layer?
----------------------------------------

OpenJPEG can create JP2 files with more than 3 components (use version 2.1.0+ 
for this), but by default, any extra components are not described
as such.  In order to do so, we need to re-wrap such an image in a
set of boxes that includes a channel definition box.  The following example
creates an ellipical mask. ::

    >>> import glymur, numpy as np
    >>> from glymur import Jp2k
    >>> rgb = Jp2k(glymur.data.goodstuff())[:]
    >>> ny, nx = rgb.shape[:2]
    >>> Y, X = np.ogrid[:ny, :nx]
    >>> mask = nx ** 2 * (Y - ny / 2) ** 2 + ny ** 2 * (X - nx / 2) ** 2 > (nx * ny / 2)**2
    >>> alpha = 255 * np.ones((ny, nx, 1), dtype=np.uint8)
    >>> alpha[mask] = 0
    >>> rgba = np.concatenate((rgb, alpha), axis=2)
    >>> jp2 = Jp2k('myfile.jp2', data=rgba)

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
    >>> cdef = glymur.jp2box.ChannelDefinitionBox(ctype, asoc)
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


**************
Basic Metadata
**************

How do I...?
============

... display metadata?
---------------------

There are two ways.  From the command line, the console script **jp2dump** is
available. ::

    $ jp2dump /path/to/glymur/installation/data/nemo.jp2

From within Python, the same result is obtained simply by printing the Jp2k
object, i.e. ::

    >>> import glymur
    >>> jp2file = glymur.data.nemo() # just a path to a JP2 file
    >>> jp2 = glymur.Jp2k(jp2file)
    >>> print(jp2)  # doctest: +SKIP
    File:  nemo.jp2
    JPEG 2000 Signature Box (jP  ) @ (0, 12)
        Signature:  0d0a870a
    File Type Box (ftyp) @ (12, 20)
        Brand:  jp2 
        Compatibility:  ['jp2 ']
    JP2 Header Box (jp2h) @ (32, 45)
        Image Header Box (ihdr) @ (40, 22)
            Size:  [1456 2592 3]
            Bitdepth:  8
            Signed:  False
            Compression:  wavelet
            Colorspace Unknown:  False
        Colour Specification Box (colr) @ (62, 15)
            Method:  enumerated colorspace
            Precedence:  0
            Colorspace:  sRGB
    Contiguous Codestream Box (jp2c) @ (77, 1132296)
        Main header:
            SOC marker segment @ (3231, 0)
            SIZ marker segment @ (3233, 47)
                Profile:  2
                Reference Grid Height, Width:  (1456 x 2592)
                Vertical, Horizontal Reference Grid Offset:  (0 x 0)
                Reference Tile Height, Width:  (1456 x 2592)
                Vertical, Horizontal Reference Tile Offset:  (0 x 0)
                Bitdepth:  (8, 8, 8)
                Signed:  (False, False, False)
                Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
            COD marker segment @ (3282, 12)
                Coding style:
                    Entropy coder, without partitions
                    SOP marker segments:  False
                    EPH marker segments:  False
                Coding style parameters:
                    Progression order:  LRCP
                    Number of layers:  2
                    Multiple component transformation usage:  reversible
                    Number of resolutions:  2
                    Code block height, width:  (64 x 64)
                    Wavelet transform:  5-3 reversible
                    Precinct size:  default, 2^15 x 2^15
                    Code block context:
                        Selective arithmetic coding bypass:  False
                        Reset context probabilities on coding pass boundaries:  False
                        Termination on each coding pass:  False
                        Vertically stripe causal context:  False
                        Predictable termination:  False
                        Segmentation symbols:  False
            QCD marker segment @ (3296, 7)
                Quantization style:  no quantization, 2 guard bits
                Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
            CME marker segment @ (3305, 37)
                "Created by OpenJPEG version 2.0.0"
     
... display less metadata?
--------------------------

The amount of metadata in a JPEG 2000 file can be overwhelming, mostly due
to the codestream and XML.  You can suppress the codestream and XML details by
making use of the :py:meth:`set_option` function::

    >>> import glymur
    >>> jpx = glymur.Jp2k(glymur.data.jpxfile())
    >>> glymur.set_option('print.codestream', False)
    >>> glymur.set_option('print.xml', False)
    >>> print(jpx)
    File:  heliov.jpx
    JPEG 2000 Signature Box (jP  ) @ (0, 12)
        Signature:  0d0a870a
    File Type Box (ftyp) @ (12, 28)
        Brand:  jpx 
        Compatibility:  ['jpx ', 'jp2 ', 'jpxb']
    JP2 Header Box (jp2h) @ (40, 847)
        Image Header Box (ihdr) @ (48, 22)
            Size:  [1024 1024 1]
            Bitdepth:  8
            Signed:  False
            Compression:  wavelet
            Colorspace Unknown:  False
        Colour Specification Box (colr) @ (70, 15)
            Method:  enumerated colorspace
            Precedence:  0
            Colorspace:  sRGB
        Palette Box (pclr) @ (85, 782)
            Size:  (256 x 3)
        Component Mapping Box (cmap) @ (867, 20)
            Component 0 ==> palette column 0
            Component 0 ==> palette column 1
            Component 0 ==> palette column 2
    Codestream Header Box (jpch) @ (887, 8)
    Compositing Layer Header Box (jplh) @ (895, 8)
    Contiguous Codestream Box (jp2c) @ (903, 313274)
    Codestream Header Box (jpch) @ (314177, 50)
        Image Header Box (ihdr) @ (314185, 22)
            Size:  [256 256 3]
            Bitdepth:  8
            Signed:  False
            Compression:  wavelet
            Colorspace Unknown:  True
        Component Mapping Box (cmap) @ (314207, 20)
            Component 0 ==> 0
            Component 1 ==> 1
            Component 2 ==> 2
    Compositing Layer Header Box (jplh) @ (314227, 31)
        Colour Group Box (cgrp) @ (314235, 23)
            Colour Specification Box (colr) @ (314243, 15)
                Method:  enumerated colorspace
                Precedence:  0
                Colorspace:  sRGB
    Contiguous Codestream Box (jp2c) @ (314258, 26609)
    Codestream Header Box (jpch) @ (340867, 42)
        Image Header Box (ihdr) @ (340875, 22)
            Size:  [4096 4096 1]
            Bitdepth:  8
            Signed:  False
            Compression:  wavelet
            Colorspace Unknown:  True
        Component Mapping Box (cmap) @ (340897, 12)
            Component 0 ==> 0
    Compositing Layer Header Box (jplh) @ (340909, 31)
        Colour Group Box (cgrp) @ (340917, 23)
            Colour Specification Box (colr) @ (340925, 15)
                Method:  enumerated colorspace
                Precedence:  0
                Colorspace:  greyscale
    Contiguous Codestream Box (jp2c) @ (340940, 1048552)
    Association Box (asoc) @ (1389492, 9579)
        Association Box (asoc) @ (1389500, 3421)
            Number List Box (nlst) @ (1389508, 16)
                Association[0]:  codestream 0
                Association[1]:  compositing layer 0
            XML Box (xml ) @ (1389524, 3397)
        Association Box (asoc) @ (1392921, 6150)
            Number List Box (nlst) @ (1392929, 16)
                Association[0]:  codestream 2
                Association[1]:  compositing layer 2
            XML Box (xml ) @ (1392945, 6126)

Now try it without suppressing the XML and codestream details.

... display the codestream in all its gory glory?
-------------------------------------------------

The codestream details are limited to the codestream header because
by default that's all the codestream metadata that is retrieved. It is, howver,
possible to print the full codestream.::

    >>> import glymur
    >>> glymur.set_option('print.codestream', True)
    >>> c = jp2.get_codestream(header_only=False)
    >>> print(c)  # doctest: +SKIP
    Codestream:
    SOC marker segment @ (3231, 0)
    SIZ marker segment @ (3233, 47)
        Profile:  no profile
        Reference Grid Height, Width:  (1456 x 2592)
        Vertical, Horizontal Reference Grid Offset:  (0 x 0)
        Reference Tile Height, Width:  (1456 x 2592)
        Vertical, Horizontal Reference Tile Offset:  (0 x 0)
        Bitdepth:  (8, 8, 8)
        Signed:  (False, False, False)
        Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
    COD marker segment @ (3282, 12)
        Coding style:
            Entropy coder, without partitions
            SOP marker segments:  False
            EPH marker segments:  False
        Coding style parameters:
            Progression order:  LRCP
            Number of layers:  2
            Multiple component transformation usage:  reversible
            Number of decomposition levels:  1
            Code block height, width:  (64 x 64)
            Wavelet transform:  5-3 reversible
            Precinct size:  (32768, 32768)
            Code block context:
                Selective arithmetic coding bypass:  False
                Reset context probabilities on coding pass boundaries:  False
                Termination on each coding pass:  False
                Vertically stripe causal context:  False
                Predictable termination:  False
                Segmentation symbols:  False
    QCD marker segment @ (3296, 7)
        Quantization style:  no quantization, 2 guard bits
        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
    CME marker segment @ (3305, 37)
        "Created by OpenJPEG version 2.0.0"
    SOT marker segment @ (3344, 10)
        Tile part index:  0
        Tile part length:  1132173
        Tile part instance:  0
        Number of tile parts:  1
    COC marker segment @ (3356, 9)
        Associated component:  1
        Coding style for this component:  Entropy coder, PARTITION = 0
        Coding style parameters:
            Number of decomposition levels:  1
            Code block height, width:  (64 x 64)
            Wavelet transform:  5-3 reversible
            Precinct size:  (32768, 32768)
            Code block context:
                Selective arithmetic coding bypass:  False
                Reset context probabilities on coding pass boundaries:  False
                Termination on each coding pass:  False
                Vertically stripe causal context:  False
                Predictable termination:  False
                Segmentation symbols:  False
    QCC marker segment @ (3367, 8)
        Associated Component:  1
        Quantization style:  no quantization, 2 guard bits
        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
    COC marker segment @ (3377, 9)
        Associated component:  2
        Coding style for this component:  Entropy coder, PARTITION = 0
        Coding style parameters:
            Number of decomposition levels:  1
            Code block height, width:  (64 x 64)
            Wavelet transform:  5-3 reversible
            Precinct size:  (32768, 32768)
            Code block context:
                Selective arithmetic coding bypass:  False
                Reset context probabilities on coding pass boundaries:  False
                Termination on each coding pass:  False
                Vertically stripe causal context:  False
                Predictable termination:  False
                Segmentation symbols:  False
    QCC marker segment @ (3388, 8)
        Associated Component:  2
        Quantization style:  no quantization, 2 guard bits
        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
    SOD marker segment @ (3398, 0)
    EOC marker segment @ (1135517, 0)


*****************
Advanced Metadata
*****************

How do I...?
============

... add XML metadata?
---------------------

You can append any number of XML boxes to a JP2 file (not to a raw codestream).
Consider the following XML file `data.xml` : ::


    >>> import glymur, io, shutil
    >>> from lxml import etree as ET
    >>> xml = io.BytesIO(b"""
    ... <info>
    ...     <locality>
    ...         <city>Boston</city>
    ...         <snowfall>24.9 inches</snowfall>
    ...     </locality>
    ...     <locality>
    ...         <city>Portland</city>
    ...         <snowfall>31.9 inches</snowfall>
    ...     </locality>
    ...     <locality>
    ...         <city>New York City</city>
    ...         <snowfall>11.4 inches</snowfall>
    ...     </locality>
    ... </info>
    ... """)
    >>> tree = ET.parse(xml)
    >>> xmlbox = glymur.jp2box.XMLBox(xml=tree)
    >>> _ = shutil.copyfile(glymur.data.nemo(), 'xml.jp2')
    >>> jp2 = glymur.Jp2k('xml.jp2')
    >>> jp2.append(xmlbox)
    >>> print(jp2)  # doctest: +SKIP

... create display and/or capture resolution boxes?
---------------------------------------------------

Capture and display resolution boxes are part of the JPEG 2000 standard.  You
may create such metadata boxes via keyword arguments.::

    >>> import glymur, numpy as np, skimage.data
    >>> data = skimage.data.camera()
    >>> vresc, hresc = 0.1, 0.2
    >>> vresd, hresd = 0.3, 0.4
    >>> j = glymur.Jp2k(
    ...     'capture.jp2',
    ...     data=skimage.data.camera(),
    ...     capture_resolution=[vresc, hresc],
    ...     display_resolution=[vresd, hresd]
    ... )
    >>> glymur.set_option('print.short', True)
    >>> print(j)
    File:  capture.jp2
    JPEG 2000 Signature Box (jP  ) @ (0, 12)
    File Type Box (ftyp) @ (12, 20)
    JP2 Header Box (jp2h) @ (32, 89)
        Image Header Box (ihdr) @ (40, 22)
        Colour Specification Box (colr) @ (62, 15)
        Resolution Box (res ) @ (77, 44)
            Capture Resolution Box (resc) @ (85, 18)
            Display Resolution Box (resd) @ (103, 18)
    Contiguous Codestream Box (jp2c) @ (121, 129606)


... reinterpret a codestream (say what)?
----------------------------------------

An existing raw codestream (or JP2 file) can be re-wrapped in a 
user-defined set of JP2 boxes.  The point to doing this might be
to provide a different interpretation of an image.  For example,
a raw codestream has no concept of a color model, whereas a JP2
file with a 3-channel codestream will by default consider that to
be an RGB image.

To get just a minimal JP2 jacket on the 
codestream provided by `goodstuff.j2k` (a file consisting of a raw codestream),
you can use the :py:meth:`wrap` method with no box argument: ::

    >>> import glymur
    >>> glymur.reset_option('all')
    >>> glymur.set_option('print.codestream', False)
    >>> jp2file = glymur.data.goodstuff()
    >>> j2k = glymur.Jp2k(jp2file)
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

The raw codestream was wrapped in a JP2 jacket with four boxes in the outer
layer (the signature, file type, JP2 header, and contiguous codestream), with
two additional boxes (image header and color specification) contained in the
JP2 header superbox.

XML boxes are not in the minimal set of box requirements for the JP2 format, so
in order to add an XML box into the mix before the codestream box, we'll need to 
re-specify all of the boxes.  If you already have a JP2 jacket in place,
you can just reuse that, though.  Take the following example content in
an XML file `favorites.xml` : ::

    >>> import glymur, io
    >>> from lxml import etree as ET
    >>> s = b"""
    ... <favorite_things>
    ...     <category>Light Ale</category>
    ... </favorite_things>
    ... """
    >>> xml = ET.parse(io.BytesIO(s))

In order to add the XML after the JP2 header box, but before the codestream box, 
the following will work. ::

    >>> boxes = jp2.box  # The box attribute is the list of JP2 boxes
    >>> xmlbox = glymur.jp2box.XMLBox(xml=xml)
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
    XML Box (xml ) @ (77, 79)
        <favorite_things>
            <category>Light Ale</category>
        </favorite_things>
    Contiguous Codestream Box (jp2c) @ (156, 115228)

As to the question of which method you should use, :py:meth:`append` or
:py:meth:`wrap`, to add metadata, you should keep in mind that :py:meth:`wrap`
produces a new JP2 file, while :py:meth:`append` modifies an existing file and
is currently limited to XML and UUID boxes.

... work with ICC profiles?
---------------------------

A detailed answer is beyond my capabilities.  What I can tell you is how to
gain access to ICC profiles that JPEG 2000 images may or may not provide for
you.  If there is an ICC profile, it will be provided in a ColourSpecification
box, but only if the :py:attr:`colorspace` attribute is None.  Here is an example
of how you can access an ICC profile in an `example JPX file
<https://github.com/uclouvain/openjpeg-data/blob/master/input/nonregression/text_GBR.jp2?raw=true>`_.
Basically what is done is that the raw bytes corresponding to the ICC profile
are wrapped in a BytesIO object, which is fed to the most-excellent Pillow package.
::

    >>> import pathlib
    >>> from glymur import Jp2k
    >>> from PIL import ImageCms
    >>> from io import BytesIO
    >>> # this assumes you have access to the test suite
    >>> p = pathlib.Path('tests/data/from-openjpeg/text_GBR.jp2')
    >>> # This next step may produce a harmless warning that has nothing to do with ICC profiles.
    >>> j = Jp2k(p)
    >>> # The 2nd sub box of the 4th box is a ColourSpecification box.
    >>> print(j.box[3].box[1].colorspace)
    None
    >>> b = BytesIO(j.box[3].box[1].icc_profile)
    >>> icc = ImageCms.ImageCmsProfile(b)

To go any further with this, you will want to consult
`the Pillow documentation <https://pillow.readthedocs.io/en/stable/>`_.
