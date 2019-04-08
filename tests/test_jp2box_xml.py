# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting the JP2 XML box layout.
"""
try:
    import importlib.resources as ir
except ImportError:
    import importlib_resources as ir
from io import BytesIO
import os
import shutil
import struct
import tempfile
import unittest
import warnings

# 3rd party library imports
try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# Local imports
import glymur
from glymur import Jp2k
from glymur.jp2box import ColourSpecificationBox, ContiguousCodestreamBox
from glymur.jp2box import FileTypeBox, ImageHeaderBox, JP2HeaderBox
from glymur.jp2box import JPEG2000SignatureBox, InvalidJp2kError

from . import fixtures, data


class TestXML(fixtures.TestCommon):
    """Test suite for XML boxes."""

    def setUp(self):
        super(TestXML, self).setUp()

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
        self.xmlfile = os.path.join(self.test_dir, 'countries.xml')
        with open(self.xmlfile, mode='wb') as tfile:
            tfile.write(raw_xml)
            tfile.flush()

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

    def test_negative_file_and_xml(self):
        """The XML should come from only one source."""
        xml_object = ET.parse(self.xmlfile)
        with self.assertRaises(InvalidJp2kError):
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

        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        xmlfile = os.path.join(self.test_dir, 'cyrillic.xml')
        with open(xmlfile, mode='wb') as tfile:
            tfile.write(xml.encode('utf-8'))
            tfile.flush()

        j2k = glymur.Jp2k(self.j2kfile)
        with open(self.temp_jp2_filename, mode='wb') as jfile:
            jp2 = j2k.wrap(jfile.name)
            xmlbox = glymur.jp2box.XMLBox(filename=xmlfile)
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
        with open(self.temp_jp2_filename, mode="wb") as ofile:
            with open(self.jp2file, mode='rb') as ifile:
                ofile.write(ifile.read())

            # Write the additional box.
            write_buffer = struct.pack('>I4s', int(1777), b'xml ')
            ofile.write(write_buffer)

            xmldata = ir.read_binary(data, 'encoding_declaration.xml')
            ofile.write(xmldata)

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
            file2 = os.path.join(self.test_dir, 'file2.jp2')
            with open(file2, mode="wb") as ofile2:
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


class TestJp2kBadXmlFile(fixtures.TestCommon):
    """Test suite for bad XML box situations"""

    @classmethod
    def setUpClass(cls):
        """Setup a JP2 file with a bad XML box.  We only need to do this once
        per class rather than once per test.
        """
        jp2file = glymur.data.nemo()
        cls.xml_test_dir = tempfile.mkdtemp()
        cls._bad_xml_file = os.path.join(cls.xml_test_dir, 'bad_xml_box.jp2')
        with open(cls._bad_xml_file, mode='wb') as tfile:
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
        shutil.rmtree(cls.xml_test_dir)

    def test_invalid_xml_box(self):
        """Should be able to recover info from xml box with bad xml."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jp2k = Jp2k(self._bad_xml_file)

        self.assertEqual(jp2k.box[3].box_id, 'xml ')
        self.assertEqual(jp2k.box[3].offset, 77)
        self.assertEqual(jp2k.box[3].length, 28)
        self.assertIsNone(jp2k.box[3].xml)


class TestBadButRecoverableXmlFile(unittest.TestCase):
    """Test suite for XML box that is bad, but we can still recover the XML."""

    @classmethod
    def setUpClass(cls):
        """Setup a JP2 file with bad bytes preceding the XML.  We only need
        to do this once per class rather than once per test.
        """
        jp2file = glymur.data.nemo()

        cls.xml_test_dir = tempfile.mkdtemp()
        cls._bad_xml_file = os.path.join(cls.xml_test_dir, 'bad_xml_box.jp2')

        with open(cls._bad_xml_file, mode='wb') as tfile:
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
