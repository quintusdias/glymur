import os
import struct
import sys
import tempfile
import uuid
import unittest
if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch
import warnings
from xml.etree import cElementTree as ET

import pkg_resources

from glymur import Jp2k
from glymur.lib import openjp2 as opj2
import glymur

lines = ['<?xpacket begin="\ufeff"'
         + '        id="W5M0MpCehiHzreSzNTczkc9d"?> '
         + '  <x:xmpmeta xmlns:x="adobe:ns:meta/"'
         + '             x:xmptk="XMP Core 4.4.0-Exiv2">'
         + '    <rdf:RDF'
         + '          xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
         + '        <rdf:Description rdf:about=""'
         + '            xmlns:dc="http://purl.org/dc/elements/1.1/">'
         + '          <dc:subject>'
         + '            <rdf:Bag>'
         + '              <rdf:li>Rubbertree</rdf:li>'
         + '            </rdf:Bag>'
         + '          </dc:subject>'
         + '        </rdf:Description>'
         + '    </rdf:RDF>\n</x:xmpmeta>'
         + '<?xpacket end="w"?>']
packet = '\n'.join(lines)

class TestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")
        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as tfile:
            cls._xmp_file = tfile.name
            with open(jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                buffer = ifile.read(77)
                tfile.write(buffer)

                # Write the uuid box with bad xml
                # Length = 4 + 4 + 16 + length of packet, id is 'xml '.
                L = 4 + 4 + 16 + len(packet.encode())
                buffer = struct.pack('>I4s', int(L), b'uuid')
                tfile.write(buffer)
                tfile.write(b'\xbe\x7a\xcf\xcb\x97\xa9\x42\xe8')
                tfile.write(b'\x9c\x71\x99\x94\x91\xe3\xaf\xac')
                tfile.write(packet.encode())

                # Get the rest of the input file.
                buffer = ifile.read()
                tfile.write(buffer)
                tfile.flush()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._xmp_file)

    def setUp(self):
        self.jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")

    def tearDown(self):
        pass

    def test_basic_xmp(self):
        b = b'\xbe\x7a\xcf\xcb\x97\xa9\x42\xe8\x9c\x71\x99\x94\x91\xe3\xaf\xac'
        xmp_uuid = uuid.UUID(bytes=b)
        jp2 = Jp2k(self._xmp_file)
        self.assertEqual(jp2.box[3].uuid, xmp_uuid)


if __name__ == "__main__":
    unittest.main()
