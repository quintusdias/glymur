# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting the JP2 XML box layout.
"""
import codecs
from io import BytesIO
import os
import pkg_resources as pkg
import struct
import sys
import tempfile
import unittest
import warnings

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import glymur
from glymur import Jp2k
from glymur.jp2box import ColourSpecificationBox, ContiguousCodestreamBox
from glymur.jp2box import FileTypeBox, ImageHeaderBox, JP2HeaderBox
from glymur.jp2box import JPEG2000SignatureBox

from . import fixtures


@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
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

    def test_bom(self):
        """
        Byte order markers are illegal in UTF-8.  Issue 185

        Original test file was input/nonregression/issue171.jp2
        """
        f = BytesIO()

        buffer = (b"<?xpacket "
                  b"begin='" + codecs.BOM_UTF8 + b"' "
                  b"id='W5M0MpCehiHzreSzNTczkc9d'?>"
                  b"<stuff>goes here</stuff>"
                  b"<?xpacket end='w'?>")

        f.write(buffer)
        num_bytes = f.tell()
        f.seek(0)

        with warnings.catch_warnings(record=True) as w:
            glymur.jp2box.XMLBox.parse(f, 0, 8 + num_bytes)
            if sys.hexversion < 0x03000000:
                assert issubclass(w[-1].category, UserWarning)
            else:
                # Python3 handles the BOM just fine.
                self.assertEqual(len(w), 0)

    def test_negative_file_and_xml(self):
        """The XML should come from only one source."""
        xml_object = ET.parse(self.xmlfile)
        with self.assertRaises((IOError, OSError)):
            glymur.jp2box.XMLBox(filename=self.xmlfile, xml=xml_object)

    def test_basic_xml(self):
        """Should be able to write a basic XMLBox"""
        j2k = Jp2k(self.j2kfile)

        self.jp2h.box = [self.ihdr, self.colr]

        doc = ET.parse(BytesIO(b'<?xml version="1.0"?><data>0</data>'))
        xmlb = glymur.jp2box.XMLBox(xml=doc)
        self.assertEqual(ET.tostring(xmlb.xml.getroot()),
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

    def test_utf8_xml_from_xml_file(self):
        """
        XMLBox from an XML file with encoding declaration.
        """
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

    def test_xml_box_with_encoding_declaration(self):
        """
        Read JP2 file with XML box having encoding declaration
        """
        with tempfile.NamedTemporaryFile(suffix=".jp2") as ofile:
            with open(self.jp2file, mode='rb') as f:
                ofile.write(f.read())

            # Write the additional box.
            write_buffer = struct.pack('>I4s', int(1777), b'xml ')
            ofile.write(write_buffer)

            relpath = os.path.join('data', 'encoding_declaration.xml')
            xml_file_path = pkg.resource_filename(__name__, relpath)

            with open(xml_file_path, 'rb') as f:
                ofile.write(f.read())

            ofile.flush()
            ofile.seek(0)

            jp2 = glymur.Jp2k(ofile.name)

            # Verify that XML box
            self.assertEqual(jp2.box[-1].box_id, 'xml ')

            namespaces = {
                'gml': "http://www.opengis.net/gml",
                'xsi': "http://www.w3.org/2001/XMLSchema-instance",
            }
            try:
                elts = jp2.box[-1].xml.xpath('//gml:rangeSet',
                                             namespaces=namespaces)
            except AttributeError:
                name = './/{{{ns}}}rangeSet'.format(ns=namespaces['gml'])
                elts = jp2.box[-1].xml.find(name)
            self.assertEqual(len(elts), 1)

            # Write it back out, read it back in.
            with tempfile.NamedTemporaryFile(suffix=".jp2") as ofile2:
                jp2_2 = jp2.wrap(ofile2.name, boxes=jp2.box)

                # Verify that XML box
                self.assertEqual(jp2_2.box[-1].box_id, 'xml ')

                try:
                    elts = jp2.box[-1].xml.xpath('//gml:rangeSet',
                                                 namespaces=namespaces)
                except AttributeError:
                    name = './/{{{ns}}}rangeSet'.format(ns=namespaces['gml'])
                    elts = jp2.box[-1].xml.find(name)

                self.assertEqual(len(elts), 1)


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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jp2k = Jp2k(self._bad_xml_file)

        self.assertEqual(jp2k.box[3].box_id, 'xml ')
        self.assertEqual(jp2k.box[3].offset, 77)
        self.assertEqual(jp2k.box[3].length, 28)
        self.assertIsNone(jp2k.box[3].xml)


@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
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

    def test_recover_from_bad_xml(self):
        """Should be able to recover info from xml box with bad xml."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jp2 = Jp2k(self._bad_xml_file)

        self.assertEqual(jp2.box[3].box_id, 'xml ')
        self.assertEqual(jp2.box[3].offset, 77)
        self.assertEqual(jp2.box[3].length, 64)
        self.assertEqual(ET.tostring(jp2.box[3].xml.getroot()),
                         b'<test>this is a test</test>')
