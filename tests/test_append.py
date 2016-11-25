"""Test suite specifically targeting JP2 box layout.
"""
# Standard library imports ...
from io import BytesIO
import shutil
import struct
import tempfile
from uuid import UUID
import unittest
try:
    # Third party library import, favored over standard library.
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# Local imports ...
import glymur
from glymur import Jp2k


class TestSuite(unittest.TestCase):
    """Tests for append method."""

    def setUp(self):
        self.j2kfile = glymur.data.goodstuff()
        self.jp2file = glymur.data.nemo()

    def test_append_xml(self):
        """Should be able to append an XML box."""
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
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
            self.assertEqual(ET.tostring(jp2.box[-1].xml.getroot()),
                             b'<data>0</data>')

    def test_only_jp2_allowed_to_append(self):
        """Only JP2 files are allowed to be appended."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            shutil.copyfile(self.j2kfile, tfile.name)

            j2k = Jp2k(tfile.name)

            # Make an XML box.  XML boxes should always be appendable to jp2
            # files.
            the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
            xmlbox = glymur.jp2box.XMLBox(xml=the_xml)
            with self.assertRaises(IOError):
                j2k.append(xmlbox)

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
            self.assertEqual(ET.tostring(jp2.box[-1].xml.getroot()),
                             b'<data>0</data>')

    def test_append_allowable_boxes(self):
        """Only XML boxes are allowed to be appended."""
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            shutil.copyfile(self.jp2file, tfile.name)

            jp2 = Jp2k(tfile.name)

            # Make a UUID box.  Only XMP UUID boxes can currently be appended.
            uuid_instance = UUID('00000000-0000-0000-0000-000000000000')
            data = b'0123456789'
            uuidbox = glymur.jp2box.UUIDBox(uuid_instance, data)
            with self.assertRaises(IOError):
                jp2.append(uuidbox)
