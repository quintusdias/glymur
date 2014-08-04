# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting JP2 box layout.
"""
# E1103:  return value from read may be list or np array
# pylint: disable=E1103

# R0902:  More than 7 instance attributes are just fine for testing.
# pylint: disable=R0902

# R0904:  Seems like pylint is fooled in this situation
# pylint: disable=R0904

# W0613:  load_tests doesn't need to use ignore or loader arguments.
# pylint: disable=W0613

import os
import struct
import sys
import tempfile
import unittest
import warnings

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import lxml.etree as ET

import glymur
from glymur import Jp2k
from glymur.jp2box import ColourSpecificationBox, ContiguousCodestreamBox
from glymur.jp2box import FileTypeBox, ImageHeaderBox, JP2HeaderBox
from glymur.jp2box import JPEG2000SignatureBox

from .fixtures import OPJ_DATA_ROOT, opj_data_file

@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
class TestXML(unittest.TestCase):
    """Test suite for XML boxes."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

        raw_xml = b"""<?xml version="1.0"?>
        <data>
            <country name="Liechtenstein">
                <rank>1</rank>
                <year>2008</year>
                <gdppc>141100</gdppc>
                <neighbor name="Austria" direction="E"/>
                <neighbor name="Switzerland" direction="W"/>
            </country>
            <country name="Singapore">
                <rank>4</rank>
                <year>2011</year>
                <gdppc>59900</gdppc>
                <neighbor name="Malaysia" direction="N"/>
            </country>
            <country name="Panama">
                <rank>68</rank>
                <year>2011</year>
                <gdppc>13600</gdppc>
                <neighbor name="Costa Rica" direction="W"/>
                <neighbor name="Colombia" direction="E"/>
            </country>
        </data>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tfile:
            tfile.write(raw_xml)
            tfile.flush()
        self.xmlfile = tfile.name

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
        self.colr = ColourSpecificationBox(colorspace=glymur.core.SRGB)

    def tearDown(self):
        os.unlink(self.xmlfile)

    def test_negative_file_and_xml(self):
        """The XML should come from only one source."""
        xml_object = ET.parse(self.xmlfile)
        with self.assertRaises((IOError, OSError)):
            glymur.jp2box.XMLBox(filename=self.xmlfile, xml=xml_object)

    def test_basic_xml(self):
        """Should be able to write a basic XMLBox"""
        j2k = Jp2k(self.j2kfile)

        self.jp2h.box = [self.ihdr, self.colr]

        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        self.assertEqual(ET.tostring(xmlb.xml),
                         b'<data>0</data>')

        boxes = [self.jp2b, self.ftyp, self.jp2h, xmlb, self.jp2c]

        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)
            jp2 = Jp2k(tfile.name)
            self.assertEqual(jp2.box[3].box_id, 'xml ')
            self.assertEqual(ET.tostring(jp2.box[3].xml.getroot()),
                             b'<data>0</data>')

    def test_xml_from_file(self):
        """Must be able to create an XML box from an XML file."""
        j2k = Jp2k(self.j2kfile)

        self.jp2h.box = [self.ihdr, self.colr]

        xmlb = glymur.jp2box.XMLBox(filename=self.xmlfile)
        boxes = [self.jp2b, self.ftyp, self.jp2h, xmlb, self.jp2c]
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            j2k.wrap(tfile.name, boxes=boxes)
            jp2 = Jp2k(tfile.name)

            output_boxes = [box.box_id for box in jp2.box]
            self.assertEqual(output_boxes, ['jP  ', 'ftyp', 'jp2h', 'xml ',
                                            'jp2c'])

            elts = jp2.box[3].xml.findall('country')
            self.assertEqual(len(elts), 3)

            neighbor = elts[1].find('neighbor')
            self.assertEqual(neighbor.attrib['name'], 'Malaysia')
            self.assertEqual(neighbor.attrib['direction'], 'N')

    def test_utf8_xml(self):
        """Should be able to write/read an XMLBox with utf-8 encoding."""
        # 'Россия' is 'Russia' in Cyrillic, not that it matters.
        xml = u"""<?xml version="1.0" encoding="utf-8"?>
        <country>Россия</country>"""
        with tempfile.NamedTemporaryFile(suffix=".xml") as xmlfile:
            xmlfile.write(xml.encode('utf-8'))
            xmlfile.flush()

            j2k = glymur.Jp2k(self.j2kfile)
            with tempfile.NamedTemporaryFile(suffix=".jp2") as jfile:
                jp2 = j2k.wrap(jfile.name)
                xmlbox = glymur.jp2box.XMLBox(filename=xmlfile.name)
                jp2.append(xmlbox)

                box_xml = jp2.box[-1].xml.getroot()
                box_xml_str = ET.tostring(box_xml,
                                          encoding='utf-8').decode('utf-8')
                self.assertEqual(box_xml_str,
                                 u'<country>Россия</country>')



@unittest.skipIf(os.name == "nt", "NamedTemporaryFile issue on windows")
class TestJp2kBadXmlFile(unittest.TestCase):
    """Test suite for bad XML box situations"""

    @classmethod
    def setUpClass(cls):
        """Setup a JP2 file with a bad XML box.  We only need to do this once
        per class rather than once per test.
        """
        jp2file = glymur.data.nemo()
        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as tfile:
            cls._bad_xml_file = tfile.name
            with open(jp2file, 'rb') as ifile:
                # Everything up until the UUID box.
                write_buffer = ifile.read(77)
                tfile.write(write_buffer)

                # Write the xml box with bad xml
                # Length = 28, id is 'xml '.
                write_buffer = struct.pack('>I4s', int(28), b'xml ')
                tfile.write(write_buffer)

                write_buffer = '<test>this is a test'
                write_buffer = write_buffer.encode()
                tfile.write(write_buffer)

                # Get the rest of the input file.
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._bad_xml_file)

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_invalid_xml_box(self):
        """Should be able to recover info from xml box with bad xml."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jp2k = Jp2k(self._bad_xml_file)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            msg = 'No XML was retrieved'
            self.assertTrue(msg in str(w[0].message))

        self.assertEqual(jp2k.box[3].box_id, 'xml ')
        self.assertEqual(jp2k.box[3].offset, 77)
        self.assertEqual(jp2k.box[3].length, 28)
        self.assertIsNone(jp2k.box[3].xml)


@unittest.skipIf(os.name == "nt", "NamedTemporaryFile issue on windows")
class TestBadButRecoverableXmlFile(unittest.TestCase):
    """Test suite for XML box that is bad, but we can still recover the XML."""

    @classmethod
    def setUpClass(cls):
        """Setup a JP2 file with bad bytes preceding the XML.  We only need
        to do this once per class rather than once per test.
        """
        jp2file = glymur.data.nemo()
        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as tfile:
            cls._bad_xml_file = tfile.name
            with open(jp2file, 'rb') as ifile:
                # Everything up until the UUID box.
                write_buffer = ifile.read(77)
                tfile.write(write_buffer)

                # Write the xml box with bad xml
                # Length = 64, id is 'xml '.
                write_buffer = struct.pack('>I4s', int(64), b'xml ')
                tfile.write(write_buffer)

                # Write out 8 bad bytes.
                write_buffer = b'\x00\x00\x07\x90xml '
                tfile.write(write_buffer)

                # Write out 48 good bytes constituting the XML payload.
                write_buffer = b'<?xml version="1.0"?>'
                tfile.write(write_buffer)
                write_buffer = b'<test>this is a test</test>'
                tfile.write(write_buffer)

                # Get the rest of the input file.
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._bad_xml_file)

    def test_bad_xml_box_warning(self):
        """Should warn in case of bad XML"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Jp2k(self._bad_xml_file)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            msg = 'A UnicodeDecodeError was encountered parsing an XML box'
            self.assertTrue(msg in str(w[0].message))

    def test_recover_from_bad_xml(self):
        """Should be able to recover info from xml box with bad xml."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jp2 = Jp2k(self._bad_xml_file)

        self.assertEqual(jp2.box[3].box_id, 'xml ')
        self.assertEqual(jp2.box[3].offset, 77)
        self.assertEqual(jp2.box[3].length, 64)
        self.assertEqual(ET.tostring(jp2.box[3].xml.getroot()),
                         b'<test>this is a test</test>')


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestXML_OpjDataRoot(unittest.TestCase):
    """Test suite for XML boxes, requires OPJ_DATA_ROOT."""

    @unittest.skipIf(sys.platform.startswith('linux'), 'Failing on linux')
    def test_bom(self):
        """Byte order markers are illegal in UTF-8.  Issue 185"""
        filename = opj_data_file(os.path.join('input',
                                              'nonregression',
                                              'issue171.jp2'))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jp2 = Jp2k(filename)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            msg = 'An illegal BOM (byte order marker) was detected and removed'
            self.assertTrue(msg in str(w[0].message))

        self.assertIsNotNone(jp2.box[3].xml)
            

    def test_invalid_utf8(self):
        """Bad byte sequence that cannot be parsed."""
        filename = opj_data_file(os.path.join('input',
                                              'nonregression',
                                              '26ccf3651020967f7778238ef5af08af.SIGFPE.d25.527.jp2'))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jp2 = Jp2k(filename)
            self.assertTrue(issubclass(w[0].category, UserWarning))

        self.assertIsNone(jp2.box[3].box[1].box[1].xml)
            


