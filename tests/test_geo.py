"""
Tests for JPEG 2000 with embedded geographic metadata.
"""
# standard library imports
import importlib.resources as ir
import io
import shutil
from unittest import skipIf
import uuid
import warnings

# 3rd party library imports
from lxml import etree as ET
import numpy as np

# local imports
from glymur.jp2box import LabelBox, XMLBox, AssociationBox, UUIDBox
import glymur
from tests import fixtures


@skipIf(not fixtures._HAVE_GDAL, 'Could not load GDAL')
class TestSuite(fixtures.TestCommon):

    def setUp(self):
        super().setUp()
        self._setUpHiRise()

    def _setUpHiRise(self):
        """
        Create a HiRISE-like jp2 file to investigate.
        """
        # Hand-create the boxes needed for HiRISE.
        the_uuid = uuid.UUID('2b0d7e97-aa2e-317d-9a33-e53161a2f7d0')
        ulst = glymur.jp2box.UUIDListBox([the_uuid])

        version = 0
        flag = [0, 0, 0]
        url = 'ESP_032436_1755_COLOR.LBL'
        debox = glymur.jp2box.DataEntryURLBox(version, flag, url)

        uuidinfo = glymur.jp2box.UUIDInfoBox([ulst, debox])

        uuid_data = ir.files('tests.data').joinpath('degenerate_geotiff.tif').read_bytes()  # noqa : E501
        the_uuid = uuid.UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03')
        geotiff_uuid = glymur.jp2box.UUIDBox(the_uuid, uuid_data)

        # Fabricate a new JP2 file out of the signature, file type, header,
        # and codestream out of nemo.jp2, but add in the UUIDInfo and UUID
        # box from HiRISE.
        jp2 = glymur.Jp2k(self.jp2file)
        boxes = [jp2.box[0], jp2.box[1], jp2.box[2], uuidinfo, geotiff_uuid,
                 jp2.box[-1]]

        self.hirise_jp2file_name = self.test_dir_path / 'hirise.jp2'
        jp2.wrap(self.hirise_jp2file_name, boxes=boxes)

    def test_gml(self):
        """
        Scenario:  construct a GML-JP2 file

        Expected Result:  gdal information is verified in the printing output
        """

        # Load the xml
        path = ir.files('tests.data.geo').joinpath('gml.xml')
        xml = ET.parse(str(path))

        # Create a file with the asoc box (child boxes are a label box and an
        # XML box with the GML).
        labelbox = LabelBox('gml.root-instance')
        xmlbox = XMLBox(xml)

        asocbox = AssociationBox([labelbox, xmlbox])
        labelbox = LabelBox('gml.data')

        asocbox = AssociationBox([labelbox, asocbox])

        # Make a copy of nemo and tack the association box onto the end of it
        dst = self.test_dir_path / 'gml.jp2'
        shutil.copyfile(self.jp2file, dst)

        j = glymur.Jp2k(dst)
        j.append(asocbox)

        actual = str(j.box[-1])

        # just verify that some gdal stuff is present
        self.assertIn('PROJCRS', actual)
        self.assertIn('BASEGEOGCRS', actual)

    def test__printing__geotiff_uuid__xml_sidecar(self):
        """
        SCENARIO:  Print a geotiff UUID with XML sidecar file.

        EXPECTED RESULT:  Should not error out.  There used to be a warning
        about GDAL not being able to print the UUID data as expected, but that
        is no longer the case.
        """
        box_data = (
            ir.files('tests.data')
              .joinpath('0220000800_uuid.dat')
              .read_bytes()
        )
        bf = io.BytesIO(box_data)
        bf.seek(8)
        box = UUIDBox.parse(bf, 0, 703)

        # Make a copy of nemo and tack the association box onto the end of it
        dst = self.test_dir_path / 'sidecar.jp2'
        shutil.copyfile(self.jp2file, dst)
        j = glymur.Jp2k(dst)
        j.append(box)

        with warnings.catch_warnings(record=True) as w:
            str(j)

        self.assertEqual(len(w), 0)

    def test_tags(self):
        """
        Scenario:  a HIRise-like JP2 has geotiff tags

        Expected Result:  the tags are verified
        """
        jp2 = glymur.Jp2k(self.hirise_jp2file_name)
        np.testing.assert_array_equal(
            jp2.box[4].data['GeoDoubleParams'],
            np.array([0.0, 180.0, 0.0, 0.0, 3396190.0, 3396190.0])
        )
        self.assertEqual(
            jp2.box[4].data['GeoAsciiParams'],
            'Equirectangular MARS|GCS_MARS|'
        )
        np.testing.assert_array_equal(
            jp2.box[4].data['GeoKeyDirectory'],
            np.array([
                1,        1,  0,    18,  # noqa
                1024,     0,  1,     1,  # noqa
                1025,     0,  1,     1,  # noqa
                1026, 34737, 21,     0,  # noqa
                2048,     0,  1, 32767,  # noqa
                2049, 34737,  9,    21,  # noqa
                2050,     0,  1, 32767,  # noqa
                2054,     0,  1,  9102,  # noqa
                2056,     0,  1, 32767,  # noqa
                2057, 34736,  1,     4,  # noqa
                2058, 34736,  1,     5,  # noqa
                3072,     0,  1, 32767,  # noqa
                3074,     0,  1, 32767,  # noqa
                3075,     0,  1,    17,  # noqa
                3076,     0,  1,  9001,  # noqa
                3082, 34736,  1,     2,  # noqa
                3083, 34736,  1,     3,  # noqa
                3088, 34736,  1,     1,  # noqa
                3089, 34736,  1,     0,  # noqa
            ])
        )
        np.testing.assert_array_equal(
            jp2.box[4].data['ModelPixelScale'],
            np.array([0.25, 0.25, 0.0])
        )
        np.testing.assert_array_equal(
            jp2.box[4].data['ModelTiePoint'],
            np.array([0.0, 0.0, 0.0, -2523306.125, -268608.875, 0.0])
        )

    def test_printing_geotiff_uuid(self):
        """
        SCENARIO:  Print a geotiff UUID.

        EXPECTED RESULT:  Should match a known geotiff UUID.  The string
        representation validates.
        """
        jp2 = glymur.Jp2k(self.hirise_jp2file_name)
        actual = str(jp2.box[4])

        # don't bother verifying the full output, just get some key parts
        self.assertIn('PROJCRS["Equirectangular MARS",', actual)

    def test_print_bad_geotiff(self):
        """
        SCENARIO:  A GeoTIFF UUID has some incorrect tags.

        EXPECTED RESULT:  No errors.  There are warnings, but they are now
        emitted directly from the TIFF library via GDAL, so they are no longer
        catchable.  No warnings.
        """
        path = ir.files('tests.data').joinpath('issue398.dat')
        with path.open('rb') as f:
            f.seek(8)
            box = glymur.jp2box.UUIDBox.parse(f, 0, 380)

        # Make a copy of nemo and tack the association box onto the end of it
        dst = self.test_dir_path / 'corrupt.jp2'
        shutil.copyfile(self.jp2file, dst)
        j = glymur.Jp2k(dst)
        j.append(box)

        with warnings.catch_warnings(record=True) as w:
            str(j.box[-1])
            self.assertEqual(len(w), 0)
