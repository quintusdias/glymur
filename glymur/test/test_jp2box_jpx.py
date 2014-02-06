# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting JPX box layout.
"""

import os
import struct
import sys
import tempfile
import unittest
import warnings
import xml.etree.cElementTree as ET

import glymur
from glymur import Jp2k


@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
class TestJPXWrap(unittest.TestCase):
    """Test suite for wrapping JPX files."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

        raw_xml = b"""<?xml version="1.0"?>
        <data>
            <country name="Liechtenstein">
                <rank>1</rank>
                <year>2008</year>
                <gdppc>141100</gdppc>
                <neighbor name="Austria" direction="E"/>
                <neighbor name="Switzerland" direction="W"/>
            </country>
        </data>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tfile:
            tfile.write(raw_xml)
            tfile.flush()
        self.xmlfile = tfile.name

    def tearDown(self):
        os.unlink(self.xmlfile)

    def test_association_box(self):
        """Wrap JP2 to JPX with asoc(nlst, xml)"""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 5]]

        # The ftyp box must be modified to jpx with jp2 compatibility.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpx ']

        numbers = (0, 1)
        nlst = glymur.jp2box.NumberListBox(numbers)
        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        asoc = glymur.jp2box.AssociationBox([nlst, xmlb])
        boxes.append(asoc)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            jpx = jp2.wrap(tfile.name, boxes=boxes)

            self.assertEqual(jpx.box[1].compatibility_list, ['jp2 ', 'jpx '])
            self.assertEqual(jpx.box[-1].box_id, 'asoc')
            self.assertEqual(jpx.box[-1].box[0].box_id, 'nlst')
            self.assertEqual(jpx.box[-1].box[1].box_id, 'xml ')
            self.assertEqual(jpx.box[-1].box[0].associations, numbers)
            self.assertEqual(ET.tostring(jpx.box[-1].box[1].xml.getroot()),
                             b'<data>0</data>')

    def test_jp2_to_jpx_sans_jp2_compatibility(self):
        """jp2 wrapped to jpx not including jp2 compatibility is wrong."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 5]]
        boxes[1].compatibility_list.append('jp2 ')
        numbers = [0, 1]
        nlst = glymur.jp2box.NumberListBox(numbers)
        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        asoc = glymur.jp2box.AssociationBox([nlst, xmlb])
        boxes.append(asoc)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(RuntimeError):
                jpx = jp2.wrap(tfile.name, boxes=boxes)

    def test_jp2_to_jpx_sans_jpx_brand(self):
        """Verify error when jp2 wrapped to jpx does not include jpx brand."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 5]]
        boxes[1].brand = 'jpx '
        numbers = [0, 1]
        nlst = glymur.jp2box.NumberListBox(numbers)
        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        asoc = glymur.jp2box.AssociationBox([nlst, xmlb])
        boxes.append(asoc)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(RuntimeError):
                jpx = jp2.wrap(tfile.name, boxes=boxes)


@unittest.skipIf(sys.hexversion < 0x03000000, "Warning assert on 2.x.")
@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
class TestJPX(unittest.TestCase):
    """Test suite for other JPX boxes."""

    def setUp(self):
        self.jpxfile = glymur.data.jpxfile()

    def tearDown(self):
        pass

    def test_rreq_box_strange_mask_length(self):
        """The standard says that the mask length should be 1, 2, 4, or 8."""
        with warnings.catch_warnings():
            # This file has a rreq mask length that we do not recognize.
            warnings.simplefilter("ignore")
            j = Jp2k(self.jpxfile)
        self.assertEqual(j.box[2].box_id, 'rreq')
        self.assertEqual(type(j.box[2]),
                         glymur.jp2box.ReaderRequirementsBox)

    def test_free_box(self):
        """Verify that we can handle a free box."""
        with warnings.catch_warnings():
            # This file has a rreq mask length that we do not recognize.
            warnings.simplefilter("ignore")
            j = Jp2k(self.jpxfile)
        self.assertEqual(j.box[16].box[0].box_id, 'free')
        self.assertEqual(type(j.box[16].box[0]), glymur.jp2box.FreeBox)

    def test_dtbl(self):
        """Verify that we can interpret Data Reference boxes."""
        # Copy the existing JPX file, add a data reference box onto the end.
        with tempfile.NamedTemporaryFile(suffix='.jpx') as tfile:
            with open(self.jpxfile, 'rb') as ifile:
                tfile.write(ifile.read())
            # 8 + 2 + 20 + 36
            boxlen = 66

            write_buffer = struct.pack('>I4s', boxlen, b'dtbl')
            tfile.write(write_buffer)

            # Just two boxes.
            write_buffer = struct.pack('>H', 2)
            tfile.write(write_buffer)

            # First data entry url box.
            # This one will have a URL with 3 null chars at the end.
            # They should be stripped off.
            write_buffer = struct.pack('>I4s', 36, b'url ')
            tfile.write(write_buffer)
            url1 = 'file:////usr/local/bin'
            write_buffer = struct.pack('>BBBB24s', 0, 0, 0, 0,
                                       (url1 + chr(0) * 3).encode())
            tfile.write(write_buffer)

            # 2nd data entry url box.
            write_buffer = struct.pack('>I4s', 20, b'url ')
            tfile.write(write_buffer)
            url2 = 'file:///'
            write_buffer = struct.pack('>BBBB8s', 0, 0, 0, 0, url2.encode())
            tfile.write(write_buffer)

            tfile.flush()

            with self.assertWarns(UserWarning):
                jpx = Jp2k(tfile.name)

            self.assertEqual(jpx.box[-1].box_id, 'dtbl')
            self.assertEqual(len(jpx.box[-1].DR), 2)
            self.assertEqual(jpx.box[-1].DR[0].url, url1)
            self.assertEqual(jpx.box[-1].DR[1].url, url2)


    def test_ftbl(self):
        """Verify that we can interpret Fragment Table boxes."""
        # Copy the existing JPX file, add a fragment table box onto the end.
        with tempfile.NamedTemporaryFile(suffix='.jpx') as tfile:
            with open(self.jpxfile, 'rb') as ifile:
                tfile.write(ifile.read())
            write_buffer = struct.pack('>I4s', 32, b'ftbl')
            tfile.write(write_buffer)

            # Just one fragment list box
            write_buffer = struct.pack('>I4s', 24, b'flst')
            tfile.write(write_buffer)

            # Simple offset, length, reference
            write_buffer = struct.pack('>HQIH', 1, 4237, 170246, 3)
            tfile.write(write_buffer)

            tfile.flush()

            with self.assertWarns(UserWarning):
                jpx = Jp2k(tfile.name)

            self.assertEqual(jpx.box[-1].box_id, 'ftbl')
            self.assertEqual(jpx.box[-1].box[0].box_id, 'flst')
            self.assertEqual(jpx.box[-1].box[0].fragment_offset, (4237,))
            self.assertEqual(jpx.box[-1].box[0].fragment_length, (170246,))
            self.assertEqual(jpx.box[-1].box[0].data_reference, (3,))

    def test_nlst(self):
        """Verify that we can handle a free box."""
        with warnings.catch_warnings():
            # This file has a rreq mask length that we do not recognize.
            warnings.simplefilter("ignore")
            j = Jp2k(self.jpxfile)
        self.assertEqual(j.box[16].box[1].box[0].box_id, 'nlst')
        self.assertEqual(type(j.box[16].box[1].box[0]),
                         glymur.jp2box.NumberListBox)

        # Two associations.
        self.assertEqual(len(j.box[16].box[1].box[0].associations), 2)

        # Codestream 0
        self.assertEqual(j.box[16].box[1].box[0].associations[0], 1 << 24)

        # Compositing Layer 0
        self.assertEqual(j.box[16].box[1].box[0].associations[1], 2 << 24)
