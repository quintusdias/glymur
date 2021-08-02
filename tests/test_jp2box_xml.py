# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting the JP2 XML box layout.
"""
# Standard library imports
import importlib.resources as ir
from io import BytesIO
import pathlib
import struct
import warnings

# 3rd party library imports
import lxml.etree as ET

# Local imports
import glymur
from glymur import Jp2k
from glymur.jp2box import (
    ColourSpecificationBox, ContiguousCodestreamBox, FileTypeBox,
    ImageHeaderBox, JP2HeaderBox
)
from glymur.jp2box import JPEG2000SignatureBox

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
        path = self.test_dir_path / 'data.xml'
        with path.open(mode='wb') as tfile:
            tfile.write(raw_xml)
            tfile.flush()
        self.xmlfile_path = path
        self.xmlfile = str(path)

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
        with self.assertRaises(RuntimeError):
            glymur.jp2box.XMLBox(filename=self.xmlfile, xml=xml_object)

    def test_basic_xml(self):
        """
        SCENARIO:  Write a JP2 file with an xml box.

        EXPECTED RESULT:  The xml box should be validated.
        """
        j2k = Jp2k(self.j2kfile)

        self.jp2h.box = [self.ihdr, self.colr]

        doc = ET.parse(BytesIO(b'<?xml version="1.0"?><data>0</data>'))
        xmlb = glymur.jp2box.XMLBox(xml=doc)
        self.assertEqual(ET.tostring(xmlb.xml.getroot()),
                         b'<data>0</data>')

        boxes = [self.jp2b, self.ftyp, self.jp2h, xmlb, self.jp2c]

        j2k.wrap(self.temp_jp2_filename, boxes=boxes)
        jp2 = Jp2k(self.temp_jp2_filename)
        self.assertEqual(jp2.box[3].box_id, 'xml ')
        self.assertEqual(ET.tostring(jp2.box[3].xml.getroot()),
                         b'<data>0</data>')

    def test_xml_from_file_as_path(self):
        """
        SCENARIO:  Create an xml box by pointing at an XML file via a pathlib
        path.

        EXPECTED RESULT:  The xml box is validated.
        """
        box = glymur.jp2box.XMLBox(filename=pathlib.Path(self.xmlfile))

        elts = box.xml.findall('country')
        self.assertEqual(len(elts), 3)

        neighbor = elts[1].find('neighbor')
        self.assertEqual(neighbor.attrib['name'], 'Malaysia')
        self.assertEqual(neighbor.attrib['direction'], 'N')

    def test_xml_from_file_as_string(self):
        """
        SCENARIO:  Create an xml box by pointing at an XML file via string

        EXPECTED RESULT:  The xml box is validated.
        """
        box = glymur.jp2box.XMLBox(filename=self.xmlfile)

        elts = box.xml.findall('country')
        self.assertEqual(len(elts), 3)

        neighbor = elts[1].find('neighbor')
        self.assertEqual(neighbor.attrib['name'], 'Malaysia')
        self.assertEqual(neighbor.attrib['direction'], 'N')

    def test_utf8_xml_from_xml_file(self):
        """
        XMLBox from an XML file with encoding declaration.
        """
        # 'Россия' is 'Russia' in Cyrillic, not that it matters.
        doc = "<country>Россия</country>"
        xml = f"""<?xml version="1.0" encoding="utf-8"?>{doc}"""

        path = self.test_dir_path / 'cyrillic.xml'
        with path.open(mode="wb") as f:
            f.write(xml.encode('utf-8'))

        xmlbox = glymur.jp2box.XMLBox(filename=str(path))

        root = xmlbox.xml.getroot()
        actual = ET.tostring(root, encoding='utf-8').decode('utf-8')
        self.assertEqual(actual, doc)

    def test_xml_box_with_encoding_declaration(self):
        """
        SCENARIO:  A JP2 file is encountered with an XML box having an encoding
        declaration.

        EXPECTED RESULT:  The xml box is validated.
        """
        xmldata = ir.read_binary(data, 'encoding_declaration.xml')
        with open(self.temp_jp2_filename, mode="wb") as ofile:
            with open(self.jp2file, mode='rb') as ifile:
                ofile.write(ifile.read())

            # Write the additional box.
            box_header = struct.pack('>I4s', len(xmldata) + 8, b'xml ')
            ofile.write(box_header)
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
        elts = jp2.box[-1].xml.xpath('//gml:rangeSet', namespaces=namespaces)
        self.assertEqual(len(elts), 1)

        # Write it back out, read it back in.
        file2 = self.test_dir_path / 'file2.jp2'
        jp2_2 = jp2.wrap(file2, boxes=jp2.box)

        # Verify that XML box
        self.assertEqual(jp2_2.box[-1].box_id, 'xml ')

        elts = jp2.box[-1].xml.xpath('//gml:rangeSet', namespaces=namespaces)

        self.assertEqual(len(elts), 1)

    def test_invalid_xml_box(self):
        """
        SCENARIO:  A JP2 file is encountered with a bad XML box.

        EXPECTED RESULT:  The XML cannot be recovered, but the JP2 parsing
        does not fail
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
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

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jp2k = Jp2k(self.temp_jp2_filename)

        self.assertEqual(jp2k.box[3].box_id, 'xml ')
        self.assertEqual(jp2k.box[3].offset, 77)
        self.assertEqual(jp2k.box[3].length, 28)
        self.assertIsNone(jp2k.box[3].xml)

    def test_recover_from_bad_xml(self):
        """
        SCENARIO:  A JP2 file is encountered with a bad XML payload, but this
        time it is recoverable error.  The XML itself is ok.

        EXPECTED RESULT:
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
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
                doc = b'<test>this is a test</test>'
                tfile.write(doc)

                # Get the rest of the input file.
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jp2 = Jp2k(self.temp_jp2_filename)

        self.assertEqual(jp2.box[3].box_id, 'xml ')
        self.assertEqual(jp2.box[3].offset, 77)
        self.assertEqual(jp2.box[3].length, 64)
        self.assertEqual(ET.tostring(jp2.box[3].xml.getroot()), doc)
