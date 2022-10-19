"""Test suite specifically targeting JP2 box layout.
"""
# Standard library imports ...
import doctest
import importlib.resources as ir
from io import BytesIO
import os
import pathlib
import re
import shutil
import struct
import tempfile
from uuid import UUID
import unittest
import warnings

# Third party library imports ...
import lxml.etree as ET
import numpy as np

# Local imports ...
import glymur
from glymur import Jp2k
from glymur.jp2box import (
    ColourSpecificationBox, ContiguousCodestreamBox, FileTypeBox,
    ImageHeaderBox, JP2HeaderBox, JPEG2000SignatureBox, BitsPerComponentBox,
    PaletteBox, UnknownBox, InvalidJp2kError
)
from glymur.core import COLOR, OPACITY, SRGB, GREYSCALE
from glymur.core import RED, GREEN, BLUE, GREY, WHOLE_IMAGE
from . import fixtures, data
from .fixtures import MetadataBase
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


def docTearDown(doctest_obj):  # pragma: no cover
    glymur.set_option('parse.full_codestream', False)


def load_tests(loader, tests, ignore):  # pragma: no cover
    """Run doc tests as well."""
    if os.name == "nt":
        # Can't do it on windows, temporary file issue.
        return tests
    tests.addTests(doctest.DocTestSuite('glymur.jp2box',
                                        tearDown=docTearDown))
    return tests


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestDataEntryURL(fixtures.TestCommon):
    """Test suite for DataEntryURL boxes."""

    def test_wrap_greyscale(self):
        """A single component should be wrapped as GREYSCALE."""
        j = Jp2k(self.jp2file)
        data = j[:]
        red = data[:, :, 0]

        # Write it back out as a raw codestream.
        file1 = self.test_dir_path / 'file1.j2k'
        j2k = glymur.Jp2k(file1, data=red)

        # Ok, now rewrap it as JP2.  The colorspace should be GREYSCALE.
        file2 = self.test_dir_path / 'file2.jp2'
        jp2 = j2k.wrap(file2)

        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_basic_url(self):
        """Just your most basic URL box."""
        # Wrap our j2k file in a JP2 box along with an interior url box.
        jp2 = Jp2k(self.jp2file)

        url = 'http://glymur.readthedocs.org'
        deurl = glymur.jp2box.DataEntryURLBox(0, (0, 0, 0), url)
        boxes = [box for box in jp2.box if box.box_id != 'uuid']
        boxes.append(deurl)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            jp22 = jp2.wrap(tfile.name, boxes=boxes)

        actdata = [box.box_id for box in jp22.box]
        expdata = ['jP  ', 'ftyp', 'jp2h', 'jp2c', 'url ']
        self.assertEqual(actdata, expdata)
        self.assertEqual(jp22.box[4].version, 0)
        self.assertEqual(jp22.box[4].flag, (0, 0, 0))
        self.assertEqual(jp22.box[4].url, url)

    def test_null_termination(self):
        """I.9.3.2 specifies that location field must be null terminated."""
        jp2 = Jp2k(self.jp2file)

        url = 'http://glymur.readthedocs.org'
        deurl = glymur.jp2box.DataEntryURLBox(0, (0, 0, 0), url)
        boxes = [box for box in jp2.box if box.box_id != 'uuid']
        boxes.append(deurl)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            jp22 = jp2.wrap(tfile.name, boxes=boxes)

            self.assertEqual(jp22.box[-1].length, 42)

            # Go to the last box.  Seek past the L, T, version,
            # and flag fields.
            with open(tfile.name, 'rb') as fptr:
                fptr.seek(jp22.box[-1].offset + 4 + 4 + 1 + 3)

                nbytes = (
                    jp22.box[-1].offset + jp22.box[-1].length - fptr.tell()
                )
                read_buffer = fptr.read(nbytes)
                read_url = read_buffer.decode('utf-8')
                self.assertEqual(url + chr(0), read_url)


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestChannelDefinition(fixtures.TestCommon):
    """Test suite for channel definition boxes."""

    @classmethod
    def setUpClass(cls):
        """Need a one_plane plane image for greyscale testing."""
        j2k = Jp2k(glymur.data.goodstuff())
        data = j2k[:]

        cls.planes_dir = pathlib.Path(tempfile.mkdtemp())

        # Write the first component back out to file.
        cls.one_plane = cls.planes_dir / 'one_plane.j2k'
        Jp2k(cls.one_plane, data=data[:, :, 0])

        # Write the first two components back out to file.
        cls.two_planes = cls.planes_dir / 'two_planes.j2k'
        Jp2k(cls.two_planes, data=data[:, :, 0:2])

        # Write four components back out to file.
        cls.four_planes = cls.planes_dir / 'four_planes.j2k'
        shape = (data.shape[0], data.shape[1], 1)
        alpha = np.zeros((shape), dtype=data.dtype)
        data4 = np.concatenate((data, alpha), axis=2)
        Jp2k(cls.four_planes, data=data4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.planes_dir)

    def setUp(self):
        super(TestChannelDefinition, self).setUp()

        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        self.jp2b = JPEG2000SignatureBox()
        self.ftyp = FileTypeBox()
        self.jp2h = JP2HeaderBox()
        self.jp2c = ContiguousCodestreamBox()
        self.ihdr = ImageHeaderBox(
            height=height, width=width, num_components=num_components
        )
        self.ihdr1 = ImageHeaderBox(
            height=height, width=width, num_components=1
        )
        self.ihdr2 = ImageHeaderBox(
            height=height, width=width, num_components=2
        )
        self.ihdr4 = ImageHeaderBox(
            height=height, width=width, num_components=4
        )
        self.colr_rgb = ColourSpecificationBox(colorspace=SRGB)
        self.colr_gr = ColourSpecificationBox(colorspace=GREYSCALE)

    def test_cdef_no_inputs(self):
        """channel_type and association are required inputs."""
        with self.assertRaises(TypeError):
            glymur.jp2box.ChannelDefinitionBox()

    def test_rgb_with_index(self):
        """Just regular RGB."""
        j2k = Jp2k(self.j2kfile)
        channel_type = [COLOR, COLOR, COLOR]
        association = [RED, GREEN, BLUE]
        cdef = glymur.jp2box.ChannelDefinitionBox(
            index=[0, 1, 2],
            channel_type=channel_type,
            association=association
        )
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2))
            self.assertEqual(
                jp2h.box[2].channel_type, (COLOR, COLOR, COLOR)
            )
            self.assertEqual(
                jp2h.box[2].association, (RED, GREEN, BLUE)
            )

    def test_rgb(self):
        """Just regular RGB, but don't supply the optional index."""
        j2k = Jp2k(self.j2kfile)
        channel_type = [COLOR, COLOR, COLOR]
        association = [RED, GREEN, BLUE]
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            j2k.wrap(tfile.name, boxes=boxes)

            jp2 = Jp2k(tfile.name)
            jp2h = jp2.box[2]
            boxes = [box.box_id for box in jp2h.box]
            self.assertEqual(boxes, ['ihdr', 'colr', 'cdef'])
            self.assertEqual(jp2h.box[2].index, (0, 1, 2))
            self.assertEqual(
                jp2h.box[2].channel_type, (COLOR, COLOR, COLOR)
            )
            self.assertEqual(
                jp2h.box[2].association, (RED, GREEN, BLUE)
            )

    def test_rgba(self):
        """Just regular RGBA."""
        j2k = Jp2k(self.four_planes)
        channel_type = (COLOR, COLOR, COLOR, OPACITY)
        association = (RED, GREEN, BLUE, WHOLE_IMAGE)
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )
        boxes = [self.ihdr4, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )
        boxes = [self.ihdr, self.colr_rgb, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_grey(self):
        """Just regular greyscale."""
        j2k = Jp2k(self.one_plane)
        channel_type = (COLOR,)
        association = (GREY,)
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )
        boxes = [self.ihdr1, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )
        boxes = [self.ihdr2, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )
        boxes = [self.ihdr, self.colr_gr, cdef]
        self.jp2h.box = boxes
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises((OSError, RuntimeError)):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_only_one_cdef_in_jp2h(self):
        """There can only be one channel definition box in the jp2 header."""
        j2k = Jp2k(self.j2kfile)

        channel_type = (COLOR, COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )

        boxes = [self.ihdr, cdef, self.colr_rgb, cdef]
        self.jp2h.box = boxes

        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]

        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_not_in_jp2h(self):
        """need cdef in jp2h"""
        j2k = Jp2k(self.j2kfile)
        boxes = [self.ihdr, self.colr_rgb]
        self.jp2h.box = boxes

        channel_type = (COLOR, COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )

        boxes = [self.jp2b, self.ftyp, self.jp2h, cdef, self.jp2c]

        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises((RuntimeError, OSError)):
                j2k.wrap(tfile.name, boxes=boxes)


class TestFileTypeBox(fixtures.TestCommon):
    """Test suite for ftyp box issues."""

    def test_bad_brand_on_parse(self):
        """
        SCENARIO:  The JP2 file file type box does not contain a valid brand.

        EXPECTED RESULT:  RuntimeError
        """
        with ir.path(data, 'issue396.jp2') as path:
            with warnings.catch_warnings():
                # Lots of things wrong with this file.
                warnings.simplefilter('ignore')
                with self.assertRaises(InvalidJp2kError):
                    Jp2k(path)

    def test_brand_unknown(self):
        """A ftyp box brand must be 'jp2 ' or 'jpx '."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ftyp = glymur.jp2box.FileTypeBox(brand='jp3')
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(InvalidJp2kError):
                ftyp.write(tfile)

    def test_cl_entry_unknown(self):
        """A ftyp box cl list can only contain 'jp2 ', 'jpx ', or 'jpxb'."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Bad compatibility list item.
            ftyp = glymur.jp2box.FileTypeBox(compatibility_list=['jp3'])
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(InvalidJp2kError):
                ftyp.write(tfile)

    def test_cl_entry_not_utf8(self):
        """A ftyp box cl list entry must be utf-8 decodable."""
        with open(self.jp2file, mode='rb') as f:
            data = f.read()

        # Replace bytes 28-32 with bad utf-8 data
        data = data[:28] + b'\xff\xff\xff\xff' + data[32:]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            tfile.write(data)
            tfile.flush()

            with self.assertWarns(UserWarning):
                Jp2k(tfile.name)


class TestResolutionBoxes(fixtures.TestCommon):
    """
    Test suite for resolution boxes
    """
    def test_repr(self):
        """
        Verify __repr__ method on resolution boxes.
        """
        resc = glymur.jp2box.CaptureResolutionBox(0.5, 2.5)
        resd = glymur.jp2box.DisplayResolutionBox(2.5, 0.5)
        res_super_box = glymur.jp2box.ResolutionBox(box=[resc, resd])

        newbox = eval(repr(res_super_box))

        self.assertEqual(newbox.box_id, 'res ')
        self.assertEqual(newbox.box[0].box_id, 'resc')
        self.assertEqual(newbox.box[0].vertical_resolution, 0.5)
        self.assertEqual(newbox.box[0].horizontal_resolution, 2.5)
        self.assertEqual(newbox.box[1].box_id, 'resd')
        self.assertEqual(newbox.box[1].vertical_resolution, 2.5)
        self.assertEqual(newbox.box[1].horizontal_resolution, 0.5)

    def test_resolution_superbox(self):
        """
        SCENARIO:  write a resolution superbox

        Expected Results:  do not error out, can parse the written box
        """
        vres = 0.5
        hres = 2.5
        resc = glymur.jp2box.CaptureResolutionBox(vres, hres)
        resd = glymur.jp2box.DisplayResolutionBox(vres, hres)
        rbox = glymur.jp2box.ResolutionBox(box=[resc, resd])

        with open(self.temp_jp2_filename, mode='wb') as tfile:
            rbox.write(tfile)

        with open(self.temp_jp2_filename, mode='rb') as tfile:
            tfile.seek(8)
            rbox_read = glymur.jp2box.ResolutionBox.parse(tfile, 0, 44)

        self.assertEqual(rbox_read.box[0].vertical_resolution, vres)
        self.assertEqual(rbox_read.box[0].horizontal_resolution, hres)
        self.assertEqual(rbox_read.box[1].vertical_resolution, vres)
        self.assertEqual(rbox_read.box[1].horizontal_resolution, hres)

    def test_write_capture_resolution_box_high_res(self):
        """
        SCENARIO:  write a capture resolution box with no information other
        than the floating point components.  The components have a very high
        resolution.

        Expected Results:  do not error out, can parse the written box
        """
        vres = 1.8738870547679375e+29
        hres = 3333444444.44444
        resc = glymur.jp2box.CaptureResolutionBox(vres, hres)

        with open(self.temp_jp2_filename, mode='wb') as tfile:
            resc.write(tfile)

        with open(self.temp_jp2_filename, mode='rb') as tfile:
            tfile.seek(8)
            resc_read = glymur.jp2box.CaptureResolutionBox.parse(tfile, 8, 18)

        np.testing.assert_allclose(
            vres, resc_read.vertical_resolution, rtol=1e-6
        )
        np.testing.assert_allclose(
            hres, resc_read.horizontal_resolution, rtol=1e-6
        )

    def test_write_capture_resolution_box_low_res(self):
        """
        SCENARIO:  write a capture resolution box with no information other
        than the floating point components.  The components have a very low
        resolution.

        Expected Results:  do not error out, can parse the written box
        """
        vres = 1.8738870547679375e-29
        hres = 0.333344444444444
        resc = glymur.jp2box.CaptureResolutionBox(vres, hres)

        with open(self.temp_jp2_filename, mode='wb') as tfile:
            resc.write(tfile)

        with open(self.temp_jp2_filename, mode='rb') as tfile:
            tfile.seek(8)
            resc_read = glymur.jp2box.CaptureResolutionBox.parse(tfile, 8, 18)

        np.testing.assert_allclose(
            vres, resc_read.vertical_resolution, rtol=1e-6
        )
        np.testing.assert_allclose(
            hres, resc_read.horizontal_resolution, rtol=1e-6
        )


class TestPaletteBox(fixtures.TestCommon):
    """Test suite for pclr box instantiation."""

    def test_writing_with_different_bitdepths(self):
        """Bitdepths must be the same when writing."""
        palette = np.array([[255, 0, 255], [0, 255, 0]], dtype=np.uint16)
        bps = (8, 16, 8)
        signed = (False, False, False)
        pclr = glymur.jp2box.PaletteBox(
            palette, bits_per_component=bps, signed=signed
        )
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(InvalidJp2kError):
                pclr.write(tfile)

    def test_signed_components(self):
        """
        Palettes with signed components are not supported.
        """
        b = BytesIO()

        # L, T
        b.write(struct.pack('>I4s', 20, b'pclr'))

        # Palette is 2 rows, 3 columns
        ncols = 3
        nrows = 2
        b.write(struct.pack('>HB', nrows, ncols))

        # bits per sample is 8, but signed
        bps = (np.int8(7), np.int8(7), np.int8(7))
        bps_signed = (x | 0x80 for x in bps)
        b.write(struct.pack('BBB', *bps_signed))

        # Write the palette itself.
        #
        buffer = np.int8([[0, 0, 0], [127, 127, 127]])
        b.write(struct.pack('BBB', *buffer[0]))
        b.write(struct.pack('BBB', *buffer[1]))

        # Seek back to point after L, T
        b.seek(8)
        with self.assertRaises(InvalidJp2kError):
            PaletteBox.parse(b, 8, 20)


class TestAppend(fixtures.TestCommon):
    """Tests for append method."""

    def test_append_xml(self):
        """Should be able to append an XML box."""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            shutil.copyfile(self.jp2file, tfile.name)

            jp2 = Jp2k(tfile.name)
            b = BytesIO(b'<?xml version="1.0"?><data>0</data>')
            doc = ET.parse(b)
            xmlbox = glymur.jp2box.XMLBox(xml=doc)
            jp2.append(xmlbox)

            # The sequence of box IDs should be the same as before, but with an
            # xml box at the end.
            box_ids = [box.box_id for box in jp2.box]
            expected = ['jP  ', 'ftyp', 'jp2h', 'uuid', 'jp2c', 'xml ']
            self.assertEqual(box_ids, expected)
            self.assertEqual(
                ET.tostring(jp2.box[-1].xml.getroot()),
                b'<data>0</data>'
            )

    def test_only_jp2_allowed_to_append(self):
        """Only JP2 files are allowed to be appended."""
        with open(self.temp_j2k_filename, mode="wb") as tfile:
            shutil.copyfile(self.j2kfile, tfile.name)

            j2k = Jp2k(tfile.name)

            # Make an XML box.  XML boxes should always be appendable to jp2
            # files.
            the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
            xmlbox = glymur.jp2box.XMLBox(xml=the_xml)
            with self.assertRaises(RuntimeError):
                j2k.append(xmlbox)

    def test_length_field_is_zero(self):
        """L=0 (length field in box header) is handled.

        L=0 implies that the containing box is the last box.  If this is not
        handled properly, the appended box is never seen.
        """
        baseline_jp2 = Jp2k(self.jp2file)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                offset = baseline_jp2.box[-1].offset
                tfile.write(ifile.read(offset))

                # Write the L, T fields of the jp2c box such that L == 0
                write_buffer = struct.pack('>I4s', int(0), b'jp2c')
                tfile.write(write_buffer)

                # Write out the rest of the codestream.
                ifile.seek(offset + 8)
                tfile.write(ifile.read())
                tfile.flush()

            jp2 = Jp2k(tfile.name)
            b = BytesIO(b'<?xml version="1.0"?><data>0</data>')
            doc = ET.parse(b)
            xmlbox = glymur.jp2box.XMLBox(xml=doc)
            jp2.append(xmlbox)

            # The sequence of box IDs should be the same as before, but with an
            # xml box at the end.
            box_ids = [box.box_id for box in jp2.box]
            expected = ['jP  ', 'ftyp', 'jp2h', 'uuid', 'jp2c', 'xml ']
            self.assertEqual(box_ids, expected)
            self.assertEqual(
                ET.tostring(jp2.box[-1].xml.getroot()),
                b'<data>0</data>'
            )

    def test_append_allowable_boxes(self):
        """Only XML boxes are allowed to be appended."""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            shutil.copyfile(self.jp2file, tfile.name)

            jp2 = Jp2k(tfile.name)

            # Make a UUID box.  Only XMP UUID boxes can currently be appended.
            uuid_instance = UUID('00000000-0000-0000-0000-000000000000')
            data = b'0123456789'
            uuidbox = glymur.jp2box.UUIDBox(uuid_instance, data)
            with self.assertRaises(RuntimeError):
                jp2.append(uuidbox)


class TestWrap(fixtures.TestCommon):
    """Tests for wrap method."""

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

    def test_wrap(self):
        """basic test for rewrapping a j2c file, no specified boxes"""
        j2k = Jp2k(self.j2kfile)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            j2k.wrap(tfile.name)
            self.verify_wrapped_raw(tfile.name)

    def test_jpx_to_jp2(self):
        """basic test for rewrapping a jpx file"""
        jpx = Jp2k(self.jpxfile)
        # Use only the signature, file type, header, and 1st codestream.
        lst = [0, 1, 2, 5]
        boxes = [jpx.box[idx] for idx in lst]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            jp2 = jpx.wrap(tfile.name, boxes=boxes)

        # Verify the outer boxes.
        boxes = [box.box_id for box in jp2.box]
        self.assertEqual(boxes, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        # Verify the inside boxes.
        boxes = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(boxes, ['ihdr', 'colr', 'pclr', 'cmap'])

        expected_offsets = [0, 12, 40, 887]
        for j, offset in enumerate(expected_offsets):
            self.assertEqual(jp2.box[j].offset, offset)

    def test_wrap_jp2(self):
        """basic test for rewrapping a jp2 file, no specified boxes"""
        j2k = Jp2k(self.jp2file)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            jp2 = j2k.wrap(tfile.name)
        boxes = [box.box_id for box in jp2.box]
        self.assertEqual(boxes, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

    def test_wrap_jp2_Lzero(self):
        """Wrap jp2 with jp2c box length is zero"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                tfile.write(ifile.read())
            # Rewrite with codestream length as zero.
            tfile.seek(3223)
            tfile.write(struct.pack('>I', 0))
            tfile.flush()

        jp = Jp2k(tfile.name)

        file2 = self.test_dir_path / 'file2.jp2'
        jp2 = jp.wrap(file2)
        boxes = [box for box in jp2.box]
        self.assertEqual(boxes[3].length, 1132296)

    def test_wrap_jp2_Lone(self):
        """Wrap jp2 with jp2c box length is 1, implies Q field"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                tfile.write(ifile.read(3223))
                # Write new L, T, Q fields
                tfile.write(struct.pack('>I4sQ', 1, b'jp2c', 1132296 + 8))
                # skip over the old L, T fields
                ifile.seek(3231)
                tfile.write(ifile.read())
            tfile.flush()
            jp2 = Jp2k(tfile.name)

        file2 = self.test_dir_path / 'file2.jp2'
        jp22 = jp2.wrap(file2)
        self.assertEqual(jp22.box[3].length, 1132296 + 8)

    def test_wrap_compatibility_not_jp2(self):
        """File type compatibility must contain jp2"""
        jp2 = Jp2k(self.jp2file)
        boxes = [box for box in jp2.box]
        boxes[1].compatibility_list = ['jpx ']
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_empty_jp2h(self):
        """JP2H box list cannot be empty."""
        jp2 = Jp2k(self.jp2file)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            boxes = jp2.box
            # Right here the jp2h superbox has two child boxes.  Empty out that
            # list to trigger the error.
            boxes[2].box = []
            with self.assertRaises(RuntimeError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_default_layout_with_boxes(self):
        """basic test for rewrapping a jp2 file, boxes specified"""
        j2k = Jp2k(self.j2kfile)
        boxes = [
            JPEG2000SignatureBox(),
            FileTypeBox(),
            JP2HeaderBox(),
            ContiguousCodestreamBox()
        ]
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)
        boxes[2].box = [
            ImageHeaderBox(
                height=height, width=width, num_components=num_components
            ),
            ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            j2k.wrap(tfile.name, boxes=boxes)
            self.verify_wrapped_raw(tfile.name)

    def test_ihdr_not_first_in_jp2h(self):
        """The specification says that ihdr must be the first box in jp2h."""
        j2k = Jp2k(self.j2kfile)
        boxes = [
            JPEG2000SignatureBox(),
            FileTypeBox(),
            JP2HeaderBox(),
            ContiguousCodestreamBox()
        ]
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)
        boxes[2].box = [
            ColourSpecificationBox(colorspace=glymur.core.SRGB),
            ImageHeaderBox(
                height=height, width=width, num_components=num_components
            )
        ]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                j2k.wrap(tfile.name, boxes=boxes)

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
        ihdr = ImageHeaderBox(
            height=height, width=width, num_components=num_components
        )
        jp2h.box = [ihdr, colr]
        boxes = [ftyp, jp2b, jp2h, jp2c]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_pclr_not_in_jp2h(self):
        """A palette box must reside in a JP2 header box."""
        palette = np.array([[255, 0, 255], [0, 255, 0]], dtype=np.int32)
        bps = (8, 8, 8)
        pclr = glymur.jp2box.PaletteBox(
            palette=palette, bits_per_component=bps, signed=(True, False, True)
        )

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
        ihdr = ImageHeaderBox(
            height=height, width=width, num_components=num_components
        )
        jp2h.box = [ihdr, colr]
        boxes = [jp2b, ftyp, jp2h, jp2c, pclr]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                j2k.wrap(tfile.name, boxes=boxes)

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
        ihdr = ImageHeaderBox(
            height=height, width=width, num_components=num_components
        )
        jp2h.box = [ihdr, colr]
        boxes = [jp2b, ftyp, jp2c, jp2h]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                j2k.wrap(tfile.name, boxes=boxes)

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
        ihdr = ImageHeaderBox(
            height=height, width=width, num_components=num_components
        )
        jp2h.box = [ihdr]
        boxes = [jp2k, ftyp, jp2h]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_wrap_jpx_to_jp2_with_unadorned_jpch(self):
        """A JPX file rewrapped with plain jpch is not allowed."""
        with open(self.temp_jp2_filename, mode='wb') as tfile1:
            jpx = Jp2k(self.jpxfile)
            boxes = [
                jpx.box[0],
                jpx.box[1],
                jpx.box[2],
                glymur.jp2box.ContiguousCodestreamBox()
            ]
            with self.assertRaises(RuntimeError):
                jpx.wrap(tfile1.name, boxes=boxes)

    def test_wrap_jpx_to_jp2_with_incorrect_jp2c_offset(self):
        """Reject A JPX file rewrapped with bad jp2c offset."""
        with open(self.temp_jp2_filename, mode='wb') as tfile1:
            jpx = Jp2k(self.jpxfile)
            jpch = jpx.box[5]

            # The offset should be 902.
            jpch.offset = 901
            jpch.length = 313274
            boxes = [jpx.box[0], jpx.box[1], jpx.box[2], jpch]
            with self.assertRaises(RuntimeError):
                jpx.wrap(tfile1.name, boxes=boxes)

    def test_wrap_jpx_to_jp2_with_correctly_specified_jp2c(self):
        """Accept A JPX file rewrapped with good jp2c."""
        with open(self.temp_jp2_filename, mode='wb') as tfile1:
            jpx = Jp2k(self.jpxfile)
            jpch = jpx.box[5]

            # This time get it right.
            jpch.offset = 903
            jpch.length = 313274
            boxes = [jpx.box[0], jpx.box[1], jpx.box[2], jpch]
            jp2 = jpx.wrap(tfile1.name, boxes=boxes)

        act_ids = [box.box_id for box in jp2.box]
        exp_ids = ['jP  ', 'ftyp', 'jp2h', 'jp2c']
        self.assertEqual(act_ids, exp_ids)

        act_offsets = [box.offset for box in jp2.box]
        exp_offsets = [0, 12, 40, 887]
        self.assertEqual(act_offsets, exp_offsets)

        act_lengths = [box.length for box in jp2.box]
        exp_lengths = [12, 28, 847, 313274]
        self.assertEqual(act_lengths, exp_lengths)

    def test_full_blown_jpx(self):
        """Rewrap a jpx file."""
        with open(self.temp_jp2_filename, mode='wb') as tfile1:
            jpx = Jp2k(self.jpxfile)
            idx = (
                list(range(5)) + list(range(9, 12)) + list(range(6, 9))
            ) + [12]
            boxes = [jpx.box[j] for j in idx]
            jpx2 = jpx.wrap(tfile1.name, boxes=boxes)
            exp_ids = [box.box_id for box in boxes]
            lengths = [box.length for box in jpx.box]
            exp_lengths = [lengths[j] for j in idx]
        act_ids = [box.box_id for box in jpx2.box]
        act_lengths = [box.length for box in jpx2.box]
        self.assertEqual(exp_ids, act_ids)
        self.assertEqual(exp_lengths, act_lengths)


class TestJp2Boxes(fixtures.TestCommon):
    """Tests for canonical JP2 boxes."""

    def test_issue588(self):
        """
        Scenario:  Construct a raw codestream box without parsing it.
        Retrieve the codestream attribute.

        Expected results:  no errors
        """
        j = Jp2k(self.jp2file)

        box = ContiguousCodestreamBox(
            main_header_offset=j.box[-1].main_header_offset,
            length=j.box[-1].length
        )
        box._filename = str(self.jp2file)
        box.codestream

    def test_no_ihdr_box(self):
        """
        SCENARIO:  The JP2/IHDR box cannot be parsed.

        EXPECTED RESULT:  An RuntimeError is issued.
        """
        # Write a new JP2 file that omits the IHDR box.
        j = Jp2k(self.jp2file)
        jp2h = [box for box in j.box if box.box_id == 'jp2h'][0]
        ihdr = jp2h.box[0]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            numbytes = ihdr.offset
            with open(self.jp2file, 'rb') as ifile:
                # Write all the way up to the ihdr box
                tfile.write(ifile.read(numbytes))

                # Seek past the ihdr box
                ifile.seek(ihdr.length, os.SEEK_CUR)

                # Write the rest of the JP2 file
                tfile.write(ifile.read(numbytes))

            tfile.flush()

            with self.assertRaises(InvalidJp2kError):
                with warnings.catch_warnings():
                    # Lots of things wrong with this file.
                    warnings.simplefilter('ignore')
                    Jp2k(tfile.name)

    def test_no_jp2c_box(self):
        """
        SCENARIO:  The JP2/JP2C box cannot be parsed.

        EXPECTED RESULT:  An RuntimeError is issued.
        """
        # Write a new JP2 file that omits the JP2C box.
        j = Jp2k(self.jp2file)
        jp2c = [box for box in j.box if box.box_id == 'jp2c'][0]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            numbytes = jp2c.offset
            with open(self.jp2file, 'rb') as ifile:
                tfile.write(ifile.read(numbytes))
            tfile.flush()

            with self.assertRaises(InvalidJp2kError):
                Jp2k(tfile.name)

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
        ihdr = glymur.jp2box.ImageHeaderBox(
            height=512, width=256, num_components=3
        )
        self.assertEqual(ihdr.height, 512)
        self.assertEqual(ihdr.width, 256)
        self.assertEqual(ihdr.num_components, 3)
        self.assertEqual(ihdr.bits_per_component, 8)
        self.assertFalse(ihdr.signed)
        self.assertFalse(ihdr.colorspace_unknown)

    def test_default_jp2headerbox(self):
        """Should be able to set jp2h boxes."""
        box = JP2HeaderBox()
        box.box = [
            ImageHeaderBox(height=512, width=256),
            ColourSpecificationBox(colorspace=glymur.core.GREYSCALE)
        ]
        self.assertTrue(True)

    def test_default_ccodestreambox(self):
        """Raw instantiation should not produce a main_header."""
        box = ContiguousCodestreamBox()
        self.assertEqual(box.box_id, 'jp2c')
        self.assertIsNone(box.codestream)

    def test_codestream_main_header_offset(self):
        """
        main_header_offset is an attribute of the ContiguousCodesStream box
        """
        j = Jp2k(self.jpxfile)
        self.assertEqual(
            j.box[5].main_header_offset, j.box[5].offset + 8
        )


class TestRepr(MetadataBase):
    """Tests for __repr__ methods."""
    def test_default_jp2k(self):
        """Should be able to eval a JPEG2000SignatureBox"""
        jp2k = glymur.jp2box.JPEG2000SignatureBox()

        # Test the representation instantiation.
        newbox = eval(repr(jp2k))
        self.assertTrue(isinstance(newbox, glymur.jp2box.JPEG2000SignatureBox))
        self.assertEqual(newbox.signature, (13, 10, 135, 10))

    def test_unknown(self):
        """Should be able to instantiate an unknown box"""
        box = UnknownBox('bpcc')

        # Test the representation instantiation.
        newbox = eval(repr(box))
        self.assertTrue(isinstance(newbox, glymur.jp2box.UnknownBox))

    def test_bpcc(self):
        """Should be able to instantiate a bpcc box"""
        bpc = (5, 5, 5, 1)
        signed = (False, False, True, False)
        box = BitsPerComponentBox(bpc, signed, length=12, offset=62)

        # Test the representation instantiation.
        newbox = eval(repr(box))
        self.assertEqual(bpc, newbox.bpc)
        self.assertEqual(signed, newbox.signed)

    def test_free(self):
        """Should be able to instantiate a free box"""
        free = glymur.jp2box.FreeBox()

        # Test the representation instantiation.
        newbox = eval(repr(free))
        self.assertTrue(isinstance(newbox, glymur.jp2box.FreeBox))

    def test_nlst(self):
        """Should be able to instantiate a number list box"""
        assn = (0, 1, 2)
        nlst = glymur.jp2box.NumberListBox(assn)

        # Test the representation instantiation.
        newbox = eval(repr(nlst))
        self.assertTrue(isinstance(newbox, glymur.jp2box.NumberListBox))
        self.assertEqual(newbox.associations, (0, 1, 2))

    def test_ftbl(self):
        """Should be able to instantiate a fragment table box"""
        flst = glymur.jp2box.FragmentListBox([89], [1132288], [0])
        ftbl = glymur.jp2box.FragmentTableBox([flst])

        # Test the representation instantiation.
        newbox = eval(repr(ftbl))
        self.assertTrue(isinstance(newbox, glymur.jp2box.FragmentTableBox))

    def test_dref(self):
        """Should be able to instantiate a data reference box"""
        dref = glymur.jp2box.DataReferenceBox()

        # Test the representation instantiation.
        newbox = eval(repr(dref))
        self.assertTrue(isinstance(newbox, glymur.jp2box.DataReferenceBox))

    def test_flst(self):
        """Should be able to instantiate a fragment list box"""
        flst = glymur.jp2box.FragmentListBox([89], [1132288], [0])

        # Test the representation instantiation.
        newbox = eval(repr(flst))
        self.assertTrue(isinstance(newbox, glymur.jp2box.FragmentListBox))
        self.assertEqual(newbox.fragment_offset, [89])
        self.assertEqual(newbox.fragment_length, [1132288])
        self.assertEqual(newbox.data_reference, [0])

    def test_default_cgrp(self):
        """Should be able to instantiate a color group box"""
        cgrp = glymur.jp2box.ColourGroupBox()

        # Test the representation instantiation.
        newbox = eval(repr(cgrp))
        self.assertTrue(isinstance(newbox, glymur.jp2box.ColourGroupBox))

    def test_default_ftyp(self):
        """Should be able to instantiate a FileTypeBox"""
        ftyp = glymur.jp2box.FileTypeBox()

        # Test the representation instantiation.
        newbox = eval(repr(ftyp))
        self.verify_filetype_box(newbox, FileTypeBox())

    def test_colourspecification_box(self):
        """Verify __repr__ method on colr box."""
        # TODO:  add icc_profile
        box = ColourSpecificationBox(colorspace=glymur.core.SRGB)

        newbox = eval(repr(box))
        self.assertEqual(newbox.method, glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(newbox.precedence, 0)
        self.assertEqual(newbox.approximation, 0)
        self.assertEqual(newbox.colorspace, glymur.core.SRGB)
        self.assertIsNone(newbox.icc_profile)

    def test_channeldefinition_box(self):
        """Verify __repr__ method on cdef box."""
        channel_type = [COLOR, COLOR, COLOR]
        association = [RED, GREEN, BLUE]
        cdef = glymur.jp2box.ChannelDefinitionBox(
            index=[0, 1, 2],
            channel_type=channel_type,
            association=association
        )
        newbox = eval(repr(cdef))
        self.assertEqual(newbox.index, (0, 1, 2))
        self.assertEqual(newbox.channel_type, (COLOR, COLOR, COLOR))
        self.assertEqual(newbox.association, (RED, GREEN, BLUE))

    def test_jp2header_box(self):
        """Verify __repr__ method on ihdr box."""
        ihdr = ImageHeaderBox(100, 200, num_components=3)
        colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        jp2h = JP2HeaderBox(box=[ihdr, colr])
        newbox = eval(repr(jp2h))
        self.assertEqual(newbox.box_id, 'jp2h')
        self.assertEqual(newbox.box[0].box_id, 'ihdr')
        self.assertEqual(newbox.box[1].box_id, 'colr')

    def test_imageheader_box(self):
        """Verify __repr__ method on jhdr box."""
        ihdr = ImageHeaderBox(100, 200, num_components=3)

        newbox = eval(repr(ihdr))
        self.assertEqual(newbox.height, 100)
        self.assertEqual(newbox.width, 200)
        self.assertEqual(newbox.num_components, 3)
        self.assertFalse(newbox.signed)
        self.assertEqual(newbox.bits_per_component, 8)
        self.assertEqual(newbox.compression, 7)
        self.assertFalse(newbox.colorspace_unknown)
        self.assertFalse(newbox.ip_provided)

    def test_association_box(self):
        """Verify __repr__ method on asoc box."""
        asoc = glymur.jp2box.AssociationBox()
        newbox = eval(repr(asoc))
        self.assertEqual(newbox.box_id, 'asoc')
        self.assertEqual(len(newbox.box), 0)

    def test_codestreamheader_box(self):
        """Verify __repr__ method on jpch box."""
        jpch = glymur.jp2box.CodestreamHeaderBox()
        newbox = eval(repr(jpch))
        self.assertEqual(newbox.box_id, 'jpch')
        self.assertEqual(len(newbox.box), 0)

    def test_compositinglayerheader_box(self):
        """Verify __repr__ method on jplh box."""
        jplh = glymur.jp2box.CompositingLayerHeaderBox()
        newbox = eval(repr(jplh))
        self.assertEqual(newbox.box_id, 'jplh')
        self.assertEqual(len(newbox.box), 0)

    def test_componentmapping_box(self):
        """Verify __repr__ method on cmap box."""
        cmap = glymur.jp2box.ComponentMappingBox(
            component_index=(0, 0, 0),
            mapping_type=(1, 1, 1),
            palette_index=(0, 1, 2)
        )
        newbox = eval(repr(cmap))
        self.assertEqual(newbox.box_id, 'cmap')
        self.assertEqual(newbox.component_index, (0, 0, 0))
        self.assertEqual(newbox.mapping_type, (1, 1, 1))
        self.assertEqual(newbox.palette_index, (0, 1, 2))

    def test_label_box(self):
        """Verify __repr__ method on label box."""
        lbl = glymur.jp2box.LabelBox("this is a test")
        newbox = eval(repr(lbl))
        self.assertEqual(newbox.box_id, 'lbl ')
        self.assertEqual(newbox.label, "this is a test")

    def test_data_entry_url_box(self):
        """Verify __repr__ method on data entry url box."""
        version = 0
        flag = (0, 0, 0)
        url = "http://readthedocs.glymur.org"
        box = glymur.jp2box.DataEntryURLBox(version, flag, url)
        newbox = eval(repr(box))
        self.assertEqual(newbox.box_id, 'url ')
        self.assertEqual(newbox.version, version)
        self.assertEqual(newbox.flag, flag)
        self.assertEqual(newbox.url, url)

    def test_uuidinfo_box(self):
        """Verify __repr__ method on uinf box."""
        uinf = glymur.jp2box.UUIDInfoBox()
        newbox = eval(repr(uinf))
        self.assertEqual(newbox.box_id, 'uinf')
        self.assertEqual(len(newbox.box), 0)

    def test_uuidlist_box(self):
        """Verify __repr__ method on ulst box."""
        uuid1 = UUID('00000000-0000-0000-0000-000000000001')
        uuid2 = UUID('00000000-0000-0000-0000-000000000002')
        uuids = [uuid1, uuid2]
        ulst = glymur.jp2box.UUIDListBox(ulst=uuids)
        newbox = eval(repr(ulst))
        self.assertEqual(newbox.box_id, 'ulst')
        self.assertEqual(newbox.ulst[0], uuid1)
        self.assertEqual(newbox.ulst[1], uuid2)

    def test_palette_box(self):
        """Verify Palette box repr."""
        palette = np.array([[255, 0, 1000], [0, 255, 0]], dtype=np.int32)
        bps = (8, 8, 16)
        box = glymur.jp2box.PaletteBox(
            palette=palette,
            bits_per_component=bps,
            signed=(True, False, True)
        )

        # Test will fail unless addition imports from numpy are done.
        from numpy import array, int32  # noqa: F401
        newbox = eval(repr(box))
        np.testing.assert_array_equal(newbox.palette, palette)
        self.assertEqual(newbox.bits_per_component, (8, 8, 16))
        self.assertEqual(newbox.signed, (True, False, True))

    def test_xml_box(self):
        """Verify xml box repr."""
        elt = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        tree = ET.ElementTree(elt)
        box = glymur.jp2box.XMLBox(xml=tree)

        pattern = r"""
            glymur.jp2box.XMLBox
            \(xml=<lxml.etree._ElementTree\sobject\sat\s0x[a-fA-F0-9]+>\)
        """
        regex = re.compile(pattern, re.VERBOSE)
        self.assertRegex(repr(box), regex)

    def test_readerrequirements_box(self):
        """Verify rreq repr method."""
        box = glymur.jp2box.ReaderRequirementsBox(
            fuam=160,
            dcm=192,
            standard_flag=(5, 61, 43),
            standard_mask=(128, 96, 64),
            vendor_feature=[],
            vendor_mask=[]
        )
        newbox = eval(repr(box))
        self.assertEqual(box.fuam, newbox.fuam)
        self.assertEqual(box.dcm, newbox.dcm)
        self.assertEqual(box.standard_flag, newbox.standard_flag)
        self.assertEqual(box.standard_mask, newbox.standard_mask)
        self.assertEqual(box.vendor_feature, newbox.vendor_feature)
        self.assertEqual(box.vendor_mask, newbox.vendor_mask)

    def test_uuid_box_generic(self):
        """Verify uuid repr method."""
        uuid_instance = UUID('00000000-0000-0000-0000-000000000000')
        data = b'0123456789'
        box = glymur.jp2box.UUIDBox(the_uuid=uuid_instance, raw_data=data)

        # Since the raw_data parameter is a sequence of bytes which could be
        # quite long, don't bother trying to make it conform to eval(repr()).
        pattern = r"""
            glymur.jp2box.UUIDBox\(
                UUID\('00000000-0000-0000-0000-000000000000'\),\s
                raw_data=<byte\sarray\s10\selements>
            \)
        """
        regex = re.compile(pattern, re.VERBOSE)
        self.assertRegex(repr(box), regex)

    def test_uuid_box_xmp(self):
        """Verify uuid repr method for XMP UUID box."""
        jp2file = glymur.data.nemo()
        j = Jp2k(jp2file)
        box = j.box[3]

        # Since the raw_data parameter is a sequence of bytes which could be
        # quite long, don't bother trying to make it conform to eval(repr()).
        pattern = r"""
            glymur.jp2box.UUIDBox\(
                UUID\('be7acfcb-97a9-42e8-9c71-999491e3afac'\),\s
                raw_data=<byte\sarray\s3122\selements>
            \)
        """
        regex = re.compile(pattern, re.VERBOSE)
        self.assertRegex(repr(box), regex)

    def test_contiguous_codestream_box(self):
        """Verify contiguous codestream box repr method."""
        jp2file = glymur.data.nemo()
        jp2 = Jp2k(jp2file)
        box = jp2.box[-1]

        # Difficult to eval(repr()) this, so just match the general pattern.
        pattern = r"""
            glymur.jp2box.ContiguousCodeStreamBox\(
                codestream=<glymur.codestream.Codestream\sobject\s
                at\s0x([a-fA-F0-9]*)>
            \)
        """
        regex = re.compile(pattern, re.VERBOSE)
        self.assertRegex(repr(box), regex)
