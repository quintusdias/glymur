"""
Test suite specifically targeting JP2 box layout.
"""
# E1103:  return value from read may be list or np array
# pylint: disable=E1103

# F0401:  unittest2 is needed on python-2.6 (pylint on 2.7)
# pylint: disable=F0401

# R0902:  More than 7 instance attributes are just fine for testing.
# pylint: disable=R0902

# R0904:  Seems like pylint is fooled in this situation
# pylint: disable=R0904

# W0613:  load_tests doesn't need to use ignore or loader arguments.
# pylint: disable=W0613

import doctest
import os
import shutil
import struct
import sys
import tempfile
import uuid
import xml.etree.cElementTree as ET

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

import numpy as np

import glymur
from glymur import Jp2k
from glymur.jp2box import ColourSpecificationBox, ContiguousCodestreamBox
from glymur.jp2box import FileTypeBox, ImageHeaderBox, JP2HeaderBox
from glymur.jp2box import JPEG2000SignatureBox
from glymur.core import COLOR, OPACITY
from glymur.core import RED, GREEN, BLUE, GREY, WHOLE_IMAGE

from .fixtures import OPENJP2_IS_V2_OFFICIAL

try:
    FORMAT_CORPUS_DATA_ROOT = os.environ['FORMAT_CORPUS_DATA_ROOT']
except KeyError:
    FORMAT_CORPUS_DATA_ROOT = None


def load_tests(loader, tests, ignore):
    """Run doc tests as well."""
    if os.name == "nt":
        # Can't do it on windows, temporary file issue.
        return tests
    tests.addTests(doctest.DocTestSuite('glymur.jp2box'))
    return tests


@unittest.skipIf(OPENJP2_IS_V2_OFFICIAL,
                 "Requires v2.0.0+ in order to run.")
@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Missing openjp2 library.")
class TestChannelDefinition(unittest.TestCase):
    """Test suite for channel definition boxes."""

    @classmethod
    def setUpClass(cls):
        """Need a one_plane plane image for greyscale testing."""
        j2k = Jp2k(glymur.data.goodstuff())
        data = j2k.read()
        # Write the first component back out to file.
        with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as tfile:
            grey_j2k = Jp2k(tfile.name, 'wb')
            grey_j2k.write(data[:, :, 0])
            cls.one_plane = tfile.name
        # Write the first two components back out to file.
        with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as tfile:
            grey_j2k = Jp2k(tfile.name, 'wb')
            grey_j2k.write(data[:, :, 0:1])
            cls.two_planes = tfile.name
        # Write four components back out to file.
        with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as tfile:
            rgba_jp2 = Jp2k(tfile.name, 'wb')
            shape = (data.shape[0], data.shape[1], 1)
            alpha = np.zeros((shape), dtype=data.dtype)
            data4 = np.concatenate((data, alpha), axis=2)
            rgba_jp2.write(data4)
            cls.four_planes = tfile.name

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.one_plane)
        os.unlink(cls.two_planes)
        os.unlink(cls.four_planes)

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        self.jp2b = JPEG2000SignatureBox()
        self.ftyp = FileTypeBox()
        self.jp2h = JP2HeaderBox()
        self.jp2c = ContiguousCodestreamBox()
        self.ihdr = ImageHeaderBox(height=height, width=width,
                                   num_components=num_components)
        self.colr_rgb = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        self.colr_gr = ColourSpecificationBox(colorspace=glymur.core.GREYSCALE)

    def tearDown(self):
        pass

    def test_cdef_no_inputs(self):
        """channel_type and association are required inputs."""
        with self.assertRaises(IOError):
            glymur.jp2box.ChannelDefinitionBox()

    def test_rgb_with_index(self):
        """Just regular RGB."""
        j2k = Jp2k(self.j2kfile)
        channel_type = [COLOR, COLOR, COLOR]
        association = [RED, GREEN, BLUE]
        cdef = glymur.jp2box.ChannelDefinitionBox(index=[0, 1, 2],
                                                  channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2))
            self.assertEqual(jp2h.box[2].channel_type,
                             (COLOR, COLOR, COLOR))
            self.assertEqual(jp2h.box[2].association,
                             (RED, GREEN, BLUE))

    def test_rgb(self):
        """Just regular RGB, but don't supply the optional index."""
        j2k = Jp2k(self.j2kfile)
        channel_type = [COLOR, COLOR, COLOR]
        association = [RED, GREEN, BLUE]
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2))
            self.assertEqual(jp2h.box[2].channel_type,
                             (COLOR, COLOR, COLOR))
            self.assertEqual(jp2h.box[2].association,
                             (RED, GREEN, BLUE))

    def test_rgba(self):
        """Just regular RGBA."""
        j2k = Jp2k(self.four_planes)
        channel_type = (COLOR, COLOR, COLOR, OPACITY)
        association = (RED, GREEN, BLUE, WHOLE_IMAGE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2, 3))
            self.assertEqual(jp2h.box[2].channel_type, channel_type)
            self.assertEqual(jp2h.box[2].association, association)

    def test_bad_rgba(self):
        """R, G, and B must be specified."""
        j2k = Jp2k(self.four_planes)
        channel_type = (COLOR, COLOR, OPACITY, OPACITY)
        association = (RED, GREEN, BLUE, WHOLE_IMAGE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_grey(self):
        """Just regular greyscale."""
        j2k = Jp2k(self.one_plane)
        channel_type = (COLOR,)
        association = (GREY,)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0,))
            self.assertEqual(jp2h.box[2].channel_type, channel_type)
            self.assertEqual(jp2h.box[2].association, association)

    def test_grey_alpha(self):
        """Just regular greyscale plus alpha."""
        j2k = Jp2k(self.two_planes)
        channel_type = (COLOR, OPACITY)
        association = (GREY, WHOLE_IMAGE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1))
            self.assertEqual(jp2h.box[2].channel_type, channel_type)
            self.assertEqual(jp2h.box[2].association, association)

    def test_bad_grey_alpha(self):
        """A greyscale image with alpha layer must specify a color channel"""
        j2k = Jp2k(self.two_planes)

        channel_type = (OPACITY, OPACITY)
        association = (GREY, WHOLE_IMAGE)

        # This cdef box
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises((OSError, IOError)):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_only_one_cdef_in_jp2h(self):
        """There can only be one channel definition box in the jp2 header."""
        j2k = Jp2k(self.j2kfile)

        channel_type = (COLOR, COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)

        boxes = [self.ihdr, cdef, self.colr_rgb, cdef]
        self.jp2h.box = boxes

        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]

        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_not_in_jp2h(self):
        """need cdef in jp2h"""
        j2k = Jp2k(self.j2kfile)
        boxes = [self.ihdr, self.colr_rgb]
        self.jp2h.box = boxes

        channel_type = (COLOR, COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)

        boxes = [self.jp2b, self.ftyp, self.jp2h, cdef, self.jp2c]

        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_bad_type(self):
        """Channel types are limited to 0, 1, 2, 65535
        Should reject if not all of index, channel_type, association the
        same length.
        """
        channel_type = (COLOR, COLOR, 3)
        association = (RED, GREEN, BLUE)
        with self.assertRaises(IOError):
            glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                               association=association)

    def test_wrong_lengths(self):
        """Should reject if not all of index, channel_type, association the
        same length.
        """
        channel_type = (COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        with self.assertRaises(IOError):
            glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                               association=association)


class TestColourSpecificationBox(unittest.TestCase):
    """Test suite for colr box instantiation."""

    def setUp(self):
        self.j2kfile = glymur.data.goodstuff()

        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        self.jp2b = JPEG2000SignatureBox()
        self.ftyp = FileTypeBox()
        self.jp2h = JP2HeaderBox()
        self.jp2c = ContiguousCodestreamBox()
        self.ihdr = ImageHeaderBox(height=height, width=width,
                                   num_components=num_components)

    def tearDown(self):
        pass

    @unittest.skipIf(os.name == "nt",
                     "Problems using NamedTemporaryFile on windows.")
    def test_colr_with_out_enum_cspace(self):
        """must supply an enumerated colorspace when writing"""
        j2k = Jp2k(self.j2kfile)

        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        boxes[2].box = [self.ihdr, ColourSpecificationBox(colorspace=None)]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(NotImplementedError):
                j2k.wrap(tfile.name, boxes=boxes)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_missing_colr_box(self):
        """jp2h must have a colr box"""
        j2k = Jp2k(self.j2kfile)
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        boxes[2].box = [self.ihdr]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_default_colr(self):
        """basic colr instantiation"""
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        self.assertEqual(colr.method,  glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(colr.precedence, 0)
        self.assertEqual(colr.approximation, 0)
        self.assertEqual(colr.colorspace, glymur.core.SRGB)
        self.assertIsNone(colr.icc_profile)

    def test_colr_with_cspace_and_icc(self):
        """Colour specification boxes can't have both."""
        with self.assertRaises((OSError, IOError)):
            colorspace = glymur.core.SRGB
            rawb = b'\x01\x02\x03\x04'
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 icc_profile=rawb)

    def test_colr_with_bad_method(self):
        """colr must have a valid method field"""
        colorspace = glymur.core.SRGB
        method = -1
        with self.assertRaises(IOError):
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 method=method)

    def test_colr_with_bad_approx(self):
        """colr must have a valid approximation field"""
        colorspace = glymur.core.SRGB
        approx = -1
        with self.assertRaises(IOError):
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 approximation=approx)


@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Missing openjp2 library.")
class TestAppend(unittest.TestCase):
    """Tests for append method."""

    def setUp(self):
        self.j2kfile = glymur.data.goodstuff()
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_append_xml(self):
        """Should be able to append an XML box."""
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            shutil.copyfile(self.jp2file, tfile.name)

            jp2 = Jp2k(tfile.name)
            the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
            xmlbox = glymur.jp2box.XMLBox(xml=the_xml)
            jp2.append(xmlbox)

            # The sequence of box IDs should be the same as before, but with an
            # xml box at the end.
            box_ids = [box.box_id for box in jp2.box]
            expected = ['jP  ', 'ftyp', 'jp2h', 'uuid', 'uuid', 'jp2c', 'xml ']
            self.assertEqual(box_ids, expected)
            self.assertEqual(ET.tostring(jp2.box[-1].xml.getroot()),
                             b'<data>0</data>')

    def test_only_jp2_allowed_to_append(self):
        """Only JP2 files are allowed to be appended."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            shutil.copyfile(self.j2kfile, tfile.name)

            jp2 = Jp2k(tfile.name)

            # Make a UUID box.
            uuid_instance = uuid.UUID('00000000-0000-0000-0000-000000000000')
            data = b'0123456789'
            uuidbox = glymur.jp2box.UUIDBox(uuid_instance, data)
            with self.assertRaises(IOError):
                jp2.append(uuidbox)

    def test_length_field_is_zero(self):
        """L=0 (length field in box header) is handled.

        L=0 implies that the containing box is the last box.  If this is not
        handled properly, the appended box is never seen.
        """
        baseline_jp2 = Jp2k(self.jp2file)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                offset = baseline_jp2.box[-1].offset
                tfile.write(ifile.read(offset))

                # Write the L, T fields of the jp2c box such that L == 0
                write_buffer = struct.pack('>I4s', int(0), b'jp2c')
                tfile.write(write_buffer)

                # Write out the rest of the codestream.
                ifile.seek(offset+8)
                tfile.write(ifile.read())
                tfile.flush()

            jp2 = Jp2k(tfile.name)
            the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
            xmlbox = glymur.jp2box.XMLBox(xml=the_xml)
            jp2.append(xmlbox)

            # The sequence of box IDs should be the same as before, but with an
            # xml box at the end.
            box_ids = [box.box_id for box in jp2.box]
            expected = ['jP  ', 'ftyp', 'jp2h', 'uuid', 'uuid', 'jp2c', 'xml ']
            self.assertEqual(box_ids, expected)
            self.assertEqual(ET.tostring(jp2.box[-1].xml.getroot()),
                             b'<data>0</data>')

    def test_only_xml_allowed_to_append(self):
        """Only XML boxes are allowed to be appended."""
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            shutil.copyfile(self.jp2file, tfile.name)

            jp2 = Jp2k(tfile.name)

            # Make a UUID box.
            uuid_instance = uuid.UUID('00000000-0000-0000-0000-000000000000')
            data = b'0123456789'
            uuidbox = glymur.jp2box.UUIDBox(uuid_instance, data)
            with self.assertRaises(IOError):
                jp2.append(uuidbox)


@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Missing openjp2 library.")
class TestWrap(unittest.TestCase):
    """Tests for wrap method."""

    def setUp(self):
        self.j2kfile = glymur.data.goodstuff()
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def verify_wrapped_raw(self, jp2file):
        """Shared fixture"""
        jp2 = Jp2k(jp2file)
        self.assertEqual(len(jp2.box), 4)

        self.assertEqual(jp2.box[0].box_id, 'jP  ')
        self.assertEqual(jp2.box[0].offset, 0)
        self.assertEqual(jp2.box[0].length, 12)
        self.assertEqual(jp2.box[0].longname, 'JPEG 2000 Signature')

        self.assertEqual(jp2.box[1].box_id, 'ftyp')
        self.assertEqual(jp2.box[1].offset, 12)
        self.assertEqual(jp2.box[1].length, 20)
        self.assertEqual(jp2.box[1].longname, 'File Type')

        self.assertEqual(jp2.box[2].box_id, 'jp2h')
        self.assertEqual(jp2.box[2].offset, 32)
        self.assertEqual(jp2.box[2].length, 45)
        self.assertEqual(jp2.box[2].longname, 'JP2 Header')

        self.assertEqual(jp2.box[3].box_id, 'jp2c')
        self.assertEqual(jp2.box[3].offset, 77)
        self.assertEqual(jp2.box[3].length, 115228)

        # jp2h super box
        self.assertEqual(len(jp2.box[2].box), 2)

        self.assertEqual(jp2.box[2].box[0].box_id, 'ihdr')
        self.assertEqual(jp2.box[2].box[0].offset, 40)
        self.assertEqual(jp2.box[2].box[0].length, 22)
        self.assertEqual(jp2.box[2].box[0].longname, 'Image Header')
        self.assertEqual(jp2.box[2].box[0].height, 800)
        self.assertEqual(jp2.box[2].box[0].width, 480)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        self.assertEqual(jp2.box[2].box[1].box_id, 'colr')
        self.assertEqual(jp2.box[2].box[1].offset, 62)
        self.assertEqual(jp2.box[2].box[1].length, 15)
        self.assertEqual(jp2.box[2].box[1].longname, 'Colour Specification')
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)
        self.assertIsNone(jp2.box[2].box[1].icc_profile)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_wrap(self):
        """basic test for rewrapping a j2c file, no specified boxes"""
        j2k = Jp2k(self.j2kfile)
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name)
            self.verify_wrapped_raw(tfile.name)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_wrap_jp2(self):
        """basic test for rewrapping a jp2 file, no specified boxes"""
        j2k = Jp2k(self.jp2file)
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            jp2 = j2k.wrap(tfile.name)
        boxes = [box.box_id for box in jp2.box]
        self.assertEqual(boxes, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_default_layout_with_boxes(self):
        """basic test for rewrapping a jp2 file, boxes specified"""
        j2k = Jp2k(self.j2kfile)
        boxes = [JPEG2000SignatureBox(),
                 FileTypeBox(),
                 JP2HeaderBox(),
                 ContiguousCodestreamBox()]
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)
        boxes[2].box = [ImageHeaderBox(height=height,
                                       width=width,
                                       num_components=num_components),
                        ColourSpecificationBox(colorspace=glymur.core.SRGB)]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)
            self.verify_wrapped_raw(tfile.name)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_ihdr_not_first_in_jp2h(self):
        """The specification says that ihdr must be the first box in jp2h."""
        j2k = Jp2k(self.j2kfile)
        boxes = [JPEG2000SignatureBox(),
                 FileTypeBox(),
                 JP2HeaderBox(),
                 ContiguousCodestreamBox()]
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)
        boxes[2].box = [ColourSpecificationBox(colorspace=glymur.core.SRGB),
                        ImageHeaderBox(height=height,
                                       width=width,
                                       num_components=num_components)]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_first_boxes_jp_and_ftyp(self):
        """first two boxes must be jP followed by ftyp"""
        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        jp2b = JPEG2000SignatureBox()
        ftyp = FileTypeBox()
        jp2h = JP2HeaderBox()
        jp2c = ContiguousCodestreamBox()
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ihdr = ImageHeaderBox(height=height, width=width,
                              num_components=num_components)
        jp2h.box = [ihdr, colr]
        boxes = [ftyp, jp2b, jp2h, jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_jp2h_not_preceeding_jp2c(self):
        """jp2h must precede jp2c"""
        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        jp2b = JPEG2000SignatureBox()
        ftyp = FileTypeBox()
        jp2h = JP2HeaderBox()
        jp2c = ContiguousCodestreamBox()
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ihdr = ImageHeaderBox(height=height, width=width,
                              num_components=num_components)
        jp2h.box = [ihdr, colr]
        boxes = [jp2b, ftyp, jp2c, jp2h]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_missing_codestream(self):
        """Need a codestream box in order to call wrap method."""
        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        jp2k = JPEG2000SignatureBox()
        ftyp = FileTypeBox()
        jp2h = JP2HeaderBox()
        ihdr = ImageHeaderBox(height=height, width=width,
                              num_components=num_components)
        jp2h.box = [ihdr]
        boxes = [jp2k, ftyp, jp2h]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)


class TestJp2Boxes(unittest.TestCase):
    """Tests for canonical JP2 boxes."""

    def test_default_jp2k(self):
        """Should be able to instantiate a JPEG2000SignatureBox"""
        jp2k = glymur.jp2box.JPEG2000SignatureBox()
        self.assertEqual(jp2k.signature, (13, 10, 135, 10))

    def test_default_ftyp(self):
        """Should be able to instantiate a FileTypeBox"""
        ftyp = glymur.jp2box.FileTypeBox()
        self.assertEqual(ftyp.brand, 'jp2 ')
        self.assertEqual(ftyp.minor_version, 0)
        self.assertEqual(ftyp.compatibility_list, ['jp2 '])

    def test_default_ihdr(self):
        """Should be able to instantiate an image header box."""
        ihdr = glymur.jp2box.ImageHeaderBox(height=512, width=256,
                                            num_components=3)
        self.assertEqual(ihdr.height,  512)
        self.assertEqual(ihdr.width,  256)
        self.assertEqual(ihdr.num_components,  3)
        self.assertEqual(ihdr.bits_per_component, 8)
        self.assertFalse(ihdr.signed)
        self.assertFalse(ihdr.colorspace_unknown)

    def test_default_jp2headerbox(self):
        """Should be able to set jp2h boxes."""
        box = JP2HeaderBox()
        box.box = [ImageHeaderBox(height=512, width=256),
                   ColourSpecificationBox(colorspace=glymur.core.GREYSCALE)]
        self.assertTrue(True)

    def test_default_ccodestreambox(self):
        """Raw instantiation should not produce a main_header."""
        box = ContiguousCodestreamBox()
        self.assertEqual(box.box_id, 'jp2c')
        self.assertIsNone(box.main_header)


class TestJpxBoxes(unittest.TestCase):
    """Tests for JPX boxes."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skipIf(FORMAT_CORPUS_DATA_ROOT is None,
                     "FORMAT_CORPUS_DATA_ROOT environment variable not set")
    def test_codestream_header(self):
        """Should recognize codestream header box."""
        jfile = os.path.join(FORMAT_CORPUS_DATA_ROOT,
                             'jp2k-formats/balloon.jpf')
        jpx = Jp2k(jfile)

        # This superbox just happens to be empty.
        self.assertEqual(jpx.box[4].box_id, 'jpch')
        self.assertEqual(len(jpx.box[4].box), 0)

    @unittest.skipIf(FORMAT_CORPUS_DATA_ROOT is None,
                     "FORMAT_CORPUS_DATA_ROOT environment variable not set")
    def test_compositing_layer_header(self):
        """Should recognize compositing layer header box."""
        jfile = os.path.join(FORMAT_CORPUS_DATA_ROOT,
                             'jp2k-formats/balloon.jpf')
        jpx = Jp2k(jfile)

        # This superbox just happens to be empty.
        self.assertEqual(jpx.box[5].box_id, 'jplh')
        self.assertEqual(len(jpx.box[5].box), 0)


@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
class TestChannelDefinition(unittest.TestCase):
    """Test suite for channel definition boxes."""

    @classmethod
    def setUpClass(cls):
        """Need a one_plane plane image for greyscale testing."""
        j2k = Jp2k(glymur.data.goodstuff())
        data = j2k.read()
        # Write the first component back out to file.
        with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as tfile:
            grey_j2k = Jp2k(tfile.name, 'wb')
            grey_j2k.write(data[:, :, 0])
            cls.one_plane = tfile.name
        # Write the first two components back out to file.
        with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as tfile:
            grey_j2k = Jp2k(tfile.name, 'wb')
            grey_j2k.write(data[:, :, 0:1])
            cls.two_planes = tfile.name
        # Write four components back out to file.
        with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as tfile:
            rgba_jp2 = Jp2k(tfile.name, 'wb')
            shape = (data.shape[0], data.shape[1], 1)
            alpha = np.zeros((shape), dtype=data.dtype)
            data4 = np.concatenate((data, alpha), axis=2)
            rgba_jp2.write(data4)
            cls.four_planes = tfile.name

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.one_plane)
        os.unlink(cls.two_planes)
        os.unlink(cls.four_planes)

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        self.jp2b = JPEG2000SignatureBox()
        self.ftyp = FileTypeBox()
        self.jp2h = JP2HeaderBox()
        self.jp2c = ContiguousCodestreamBox()
        self.ihdr = ImageHeaderBox(height=height, width=width,
                                   num_components=num_components)
        self.colr_rgb = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        self.colr_gr = ColourSpecificationBox(colorspace=glymur.core.GREYSCALE)

    def tearDown(self):
        pass

    def test_cdef_no_inputs(self):
        """channel_type and association are required inputs."""
        with self.assertRaises(IOError):
            glymur.jp2box.ChannelDefinitionBox()

    def test_rgb_with_index(self):
        """Just regular RGB."""
        j2k = Jp2k(self.j2kfile)
        channel_type = [COLOR, COLOR, COLOR]
        association = [RED, GREEN, BLUE]
        cdef = glymur.jp2box.ChannelDefinitionBox(index=[0, 1, 2],
                                                  channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2))
            self.assertEqual(jp2h.box[2].channel_type,
                             (COLOR, COLOR, COLOR))
            self.assertEqual(jp2h.box[2].association,
                             (RED, GREEN, BLUE))

    def test_rgb(self):
        """Just regular RGB, but don't supply the optional index."""
        j2k = Jp2k(self.j2kfile)
        channel_type = [COLOR, COLOR, COLOR]
        association = [RED, GREEN, BLUE]
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2))
            self.assertEqual(jp2h.box[2].channel_type,
                             (COLOR, COLOR, COLOR))
            self.assertEqual(jp2h.box[2].association,
                             (RED, GREEN, BLUE))

    def test_rgba(self):
        """Just regular RGBA."""
        j2k = Jp2k(self.four_planes)
        channel_type = (COLOR, COLOR, COLOR, OPACITY)
        association = (RED, GREEN, BLUE, WHOLE_IMAGE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2, 3))
            self.assertEqual(jp2h.box[2].channel_type, channel_type)
            self.assertEqual(jp2h.box[2].association, association)

    def test_bad_rgba(self):
        """R, G, and B must be specified."""
        j2k = Jp2k(self.four_planes)
        channel_type = (COLOR, COLOR, OPACITY, OPACITY)
        association = (RED, GREEN, BLUE, WHOLE_IMAGE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_grey(self):
        """Just regular greyscale."""
        j2k = Jp2k(self.one_plane)
        channel_type = (COLOR,)
        association = (GREY,)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0,))
            self.assertEqual(jp2h.box[2].channel_type, channel_type)
            self.assertEqual(jp2h.box[2].association, association)

    def test_grey_alpha(self):
        """Just regular greyscale plus alpha."""
        j2k = Jp2k(self.two_planes)
        channel_type = (COLOR, OPACITY)
        association = (GREY, WHOLE_IMAGE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1))
            self.assertEqual(jp2h.box[2].channel_type, channel_type)
            self.assertEqual(jp2h.box[2].association, association)

    def test_bad_grey_alpha(self):
        """A greyscale image with alpha layer must specify a color channel"""
        j2k = Jp2k(self.two_planes)

        channel_type = (OPACITY, OPACITY)
        association = (GREY, WHOLE_IMAGE)

        # This cdef box
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        boxes = [self.ihdr, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises((OSError, IOError)):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_only_one_cdef_in_jp2h(self):
        """There can only be one channel definition box in the jp2 header."""
        j2k = Jp2k(self.j2kfile)

        channel_type = (COLOR, COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)

        boxes = [self.ihdr, cdef, self.colr_rgb, cdef]
        self.jp2h.box = boxes

        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]

        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_not_in_jp2h(self):
        """need cdef in jp2h"""
        j2k = Jp2k(self.j2kfile)
        boxes = [self.ihdr, self.colr_rgb]
        self.jp2h.box = boxes

        channel_type = (COLOR, COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)

        boxes = [self.jp2b, self.ftyp, self.jp2h, cdef, self.jp2c]

        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_bad_type(self):
        """Channel types are limited to 0, 1, 2, 65535
        Should reject if not all of index, channel_type, association the
        same length.
        """
        channel_type = (COLOR, COLOR, 3)
        association = (RED, GREEN, BLUE)
        with self.assertRaises(IOError):
            glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                               association=association)

    def test_wrong_lengths(self):
        """Should reject if not all of index, channel_type, association the
        same length.
        """
        channel_type = (COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        with self.assertRaises(IOError):
            glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                               association=association)


if __name__ == "__main__":
    unittest.main()
