import doctest
import tempfile
import unittest

import numpy as np
import pkg_resources

import glymur
from glymur import Jp2k
from glymur.jp2box import *


# Doc tests should be run as well.
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite('glymur.jp2box'))
    return tests


@unittest.skipIf(glymur.lib.openjp2._OPENJP2 is None,
                 "Missing openjp2 library.")
class TestJp2Boxes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # We need a raw codestream, so use the one in nemo.jp2.
        jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                  "data/nemo.jp2")
        j = Jp2k(jp2file)
        c = [box for box in j.box if box.id == 'jp2c'][0]

        with tempfile.NamedTemporaryFile(suffix='.j2c', delete=False) as ofile:
            with open(jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                ifile.seek(c.offset+8)
                ofile.write(ifile.read(c.length))

        cls.raw_codestream = ofile.name

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.raw_codestream)

    def setUp(self):
        self.jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")

    def tearDown(self):
        pass

    def test_default_JPEG2000SignatureBox(self):
        # Should be able to instantiate a JPEG2000SignatureBox
        b = glymur.jp2box.JPEG2000SignatureBox()
        self.assertEqual(b.signature, (13, 10, 135, 10))

    def test_default_FileTypeBox(self):
        # Should be able to instantiate a FileTypeBox
        b = glymur.jp2box.FileTypeBox()
        self.assertEqual(b.brand, 'jp2 ')
        self.assertEqual(b.minor_version, 0)
        self.assertEqual(b.compatibility_list, ['jp2 '])

    def test_default_ImageHeaderBox(self):
        # Should be able to instantiate an image header box.
        b = glymur.jp2box.ImageHeaderBox(height=512, width=256,
                                         num_components=3)
        self.assertEqual(b.height,  512)
        self.assertEqual(b.width,  256)
        self.assertEqual(b.num_components,  3)
        self.assertEqual(b.bits_per_component, 8)
        self.assertFalse(b.signed)
        self.assertFalse(b.colorspace_unknown)

    def test_default_ColourSpecificationBox(self):
        b = glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.SRGB)
        self.assertEqual(b.method,  glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(b.precedence, 0)
        self.assertEqual(b.approximation, 0)
        self.assertEqual(b.colorspace, glymur.core.SRGB)
        self.assertIsNone(b.icc_profile)

    def test_ColourSpecificationBox_with_colorspace_and_icc(self):
        # Colour specification boxes can't have both.
        with self.assertRaises(IOError):
            colorspace = glymur.core.SRGB
            icc_profile = b'\x01\x02\x03\x04'
            b = glymur.jp2box.ColourSpecificationBox(colorspace, icc_profile)

    def test_ColourSpecificationBox_with_bad_method(self):
        colorspace = glymur.core.SRGB
        method = -1
        with self.assertRaises(IOError):
            b = glymur.jp2box.ColourSpecificationBox(colorspace, method)

    def test_ColourSpecificationBox_with_bad_approximation(self):
        colorspace = glymur.core.SRGB
        approximation = -1
        with self.assertRaises(IOError):
            b = glymur.jp2box.ColourSpecificationBox(colorspace, approximation)

    def test_default_JP2HeaderBox(self):
        b1 = JP2HeaderBox()
        b1.box = [ImageHeaderBox(height=512, width=256),
                  ColourSpecificationBox(colorspace=glymur.core.GREYSCALE)]

    def test_default_ContiguousCodestreamBox(self):
        b = ContiguousCodestreamBox()
        self.assertEqual(b.id, 'jp2c')
        self.assertEqual(b.main_header, [])

    def verify_wrapped_raw(self, jp2file):
        jp2 = Jp2k(jp2file) 
        self.assertEqual(len(jp2.box), 4)

        self.assertEqual(jp2.box[0].id, 'jP  ')
        self.assertEqual(jp2.box[0].offset, 0)
        self.assertEqual(jp2.box[0].length, 12)
        self.assertEqual(jp2.box[0].longname, 'JPEG 2000 Signature')

        self.assertEqual(jp2.box[1].id, 'ftyp')
        self.assertEqual(jp2.box[1].offset, 12)
        self.assertEqual(jp2.box[1].length, 20)
        self.assertEqual(jp2.box[1].longname, 'File Type')

        self.assertEqual(jp2.box[2].id, 'jp2h')
        self.assertEqual(jp2.box[2].offset, 32)
        self.assertEqual(jp2.box[2].length, 45)
        self.assertEqual(jp2.box[2].longname, 'JP2 Header')

        self.assertEqual(jp2.box[3].id, 'jp2c')
        self.assertEqual(jp2.box[3].offset, 77)
        self.assertEqual(jp2.box[3].length, 1133427)

        # jp2h super box
        self.assertEqual(len(jp2.box[2].box), 2)

        self.assertEqual(jp2.box[2].box[0].id, 'ihdr')
        self.assertEqual(jp2.box[2].box[0].offset, 40)
        self.assertEqual(jp2.box[2].box[0].length, 22)
        self.assertEqual(jp2.box[2].box[0].longname, 'Image Header')
        self.assertEqual(jp2.box[2].box[0].height, 1456)
        self.assertEqual(jp2.box[2].box[0].width, 2592)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        self.assertEqual(jp2.box[2].box[1].id, 'colr')
        self.assertEqual(jp2.box[2].box[1].offset, 62)
        self.assertEqual(jp2.box[2].box[1].length, 15)
        self.assertEqual(jp2.box[2].box[1].longname, 'Colour Specification')
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)
        self.assertIsNone(jp2.box[2].box[1].icc_profile)

    def test_wrap(self):
        j2k = Jp2k(self.raw_codestream)
        with tempfile.NamedTemporaryFile(suffix=".jp2", mode="wb") as tfile:
            j2k.wrap(tfile.name)
            self.verify_wrapped_raw(tfile.name)

    def test_default_layout_but_with_specified_boxes(self):
        j2k = Jp2k(self.raw_codestream)
        boxes = [JPEG2000SignatureBox(),
                 FileTypeBox(),
                 JP2HeaderBox(),
                 ContiguousCodestreamBox()]
        c = j2k.get_codestream()
        height = c.segment[1].Ysiz
        width = c.segment[1].Xsiz
        num_components = len(c.segment[1].XRsiz)
        boxes[2].box = [ImageHeaderBox(height=height,
                                       width=width,
                                       num_components=num_components),
                        ColourSpecificationBox(colorspace=glymur.core.SRGB)]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)
            self.verify_wrapped_raw(tfile.name)

    def test_image_header_box_not_first_in_jp2_header(self):
        # The specification says that ihdr must be the first box in jp2h.
        j2k = Jp2k(self.raw_codestream)
        boxes = [JPEG2000SignatureBox(),
                 FileTypeBox(),
                 JP2HeaderBox(),
                 ContiguousCodestreamBox()]
        c = j2k.get_codestream()
        height = c.segment[1].Ysiz
        width = c.segment[1].Xsiz
        num_components = len(c.segment[1].XRsiz)
        boxes[2].box = [ColourSpecificationBox(colorspace=glymur.core.SRGB),
                        ImageHeaderBox(height=height,
                                       width=width,
                                       num_components=num_components)]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)


    def test_color_specification_box_with_out_enumerated_colorspace(self):
        j2k = Jp2k(self.raw_codestream)
        boxes = [JPEG2000SignatureBox(),
                 FileTypeBox(),
                 JP2HeaderBox(),
                 ContiguousCodestreamBox()]
        c = j2k.get_codestream()
        height = c.segment[1].Ysiz
        width = c.segment[1].Xsiz
        num_components = len(c.segment[1].XRsiz)
        boxes[2].box = [ImageHeaderBox(height=height,
                                       width=width,
                                       num_components=num_components),
                        ColourSpecificationBox(colorspace=None)]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(NotImplementedError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_default_xml(self):
        # Should be able to write an xml box.
        self.assertTrue(False)

    def test_default_component_definition(self):
        # Should be able to specify a component definition box in order to,
        # say, create an image with an alpha layer.
        self.assertTrue(False)

    def test_first_2_boxes_not_jP_and_ftyp(self):
        j2k = Jp2k(self.raw_codestream)
        c = j2k.get_codestream()
        height = c.segment[1].Ysiz
        width = c.segment[1].Xsiz
        num_components = len(c.segment[1].XRsiz)

        jP = JPEG2000SignatureBox()
        ftyp = FileTypeBox()
        jp2h = JP2HeaderBox()
        jp2c = ContiguousCodestreamBox()
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ihdr = ImageHeaderBox(height=height, width=width,
                              num_components=num_components)
        jp2h.box = [ihdr, colr]
        boxes = [ftyp, jP, jp2h, jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_jp2h_not_preceeding_jp2c(self):
        j2k = Jp2k(self.raw_codestream)
        c = j2k.get_codestream()
        height = c.segment[1].Ysiz
        width = c.segment[1].Xsiz
        num_components = len(c.segment[1].XRsiz)

        jP = JPEG2000SignatureBox()
        ftyp = FileTypeBox()
        jp2h = JP2HeaderBox()
        jp2c = ContiguousCodestreamBox()
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ihdr = ImageHeaderBox(height=height, width=width,
                              num_components=num_components)
        jp2h.box = [ihdr, colr]
        boxes = [jP, ftyp, jp2c, jp2h]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_ihdr_not_first_in_jp2h(self):
        j2k = Jp2k(self.raw_codestream)
        c = j2k.get_codestream()
        height = c.segment[1].Ysiz
        width = c.segment[1].Xsiz
        num_components = len(c.segment[1].XRsiz)

        jP = JPEG2000SignatureBox()
        ftyp = FileTypeBox()
        jp2h = JP2HeaderBox()
        jp2c = ContiguousCodestreamBox()
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ihdr = ImageHeaderBox(height=height, width=width,
                              num_components=num_components)
        jp2h.box = [colr, ihdr]
        boxes = [jP, ftyp, jp2h, jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_colr_box_not_in_jp2h(self):
        j2k = Jp2k(self.raw_codestream)
        c = j2k.get_codestream()
        height = c.segment[1].Ysiz
        width = c.segment[1].Xsiz
        num_components = len(c.segment[1].XRsiz)

        jP = JPEG2000SignatureBox()
        ftyp = FileTypeBox()
        jp2h = JP2HeaderBox()
        jp2c = ContiguousCodestreamBox()
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ihdr = ImageHeaderBox(height=height, width=width,
                              num_components=num_components)
        jp2h.box = [ihdr]
        boxes = [jP, ftyp, jp2h, jp2c, colr]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)


if __name__ == "__main__":
    unittest.main()
