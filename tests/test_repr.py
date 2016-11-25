"""Test suite specifically targeting JP2 box layout.
"""
# Standard library imports ...
import sys
from uuid import UUID
import unittest
try:
    # Third party library import, favored over standard library.
    import lxml.etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# Third party library imports ...
import numpy as np

# Local imports ...
import glymur
from glymur import Jp2k
from glymur.jp2box import ColourSpecificationBox
from glymur.jp2box import ImageHeaderBox, JP2HeaderBox
from glymur.jp2box import BitsPerComponentBox, UnknownBox
from glymur.core import COLOR, RED, GREEN, BLUE
from .fixtures import MetadataBase


class TestSuite(MetadataBase):
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
        expected = glymur.jp2box.FileTypeBox()

        # Test the representation instantiation.
        actual = eval(repr(expected))

        self.assertEqual(actual.brand, expected.brand)
        self.assertEqual(actual.minor_version, expected.minor_version)
        self.assertEqual(actual.minor_version, 0)
        for cl in expected.compatibility_list:
            self.assertIn(cl, actual.compatibility_list)

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
        cdef = glymur.jp2box.ChannelDefinitionBox(index=[0, 1, 2],
                                                  channel_type=channel_type,
                                                  association=association)
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
        cmap = glymur.jp2box.ComponentMappingBox(component_index=(0, 0, 0),
                                                 mapping_type=(1, 1, 1),
                                                 palette_index=(0, 1, 2))
        newbox = eval(repr(cmap))
        self.assertEqual(newbox.box_id, 'cmap')
        self.assertEqual(newbox.component_index, (0, 0, 0))
        self.assertEqual(newbox.mapping_type, (1, 1, 1))
        self.assertEqual(newbox.palette_index, (0, 1, 2))

    def test_resolution_boxes(self):
        """Verify __repr__ method on resolution boxes."""
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
        box = glymur.jp2box.PaletteBox(palette=palette, bits_per_component=bps,
                                       signed=(True, False, True))

        # Test will fail unless addition imports from numpy are done.
        from numpy import array, int32
        newbox = eval(repr(box))
        np.testing.assert_array_equal(newbox.palette, palette)
        self.assertEqual(newbox.bits_per_component, (8, 8, 16))
        self.assertEqual(newbox.signed, (True, False, True))

    @unittest.skipIf('lxml' not in sys.modules.keys(), "No lxml")
    def test_xml_box(self):
        """Verify xml box repr."""
        elt = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        tree = ET.ElementTree(elt)
        box = glymur.jp2box.XMLBox(xml=tree)

        regexp = r"""glymur.jp2box.XMLBox"""
        regexp += r"""[(]xml=<lxml.etree._ElementTree\sobject\s"""
        regexp += """at\s0x([a-fA-F0-9]*)>[)]"""

        if sys.hexversion < 0x03000000:
            self.assertRegexpMatches(repr(box), regexp)
        else:
            self.assertRegex(repr(box), regexp)

    def test_readerrequirements_box(self):
        """Verify rreq repr method."""
        box = glymur.jp2box.ReaderRequirementsBox(fuam=160, dcm=192,
                                                  standard_flag=(5, 61, 43),
                                                  standard_mask=(128, 96, 64),
                                                  vendor_feature=[],
                                                  vendor_mask=[])
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
        regexp = r"""glymur.jp2box.UUIDBox\("""
        regexp += """the_uuid="""
        regexp += """UUID\('00000000-0000-0000-0000-000000000000'\),\s"""
        regexp += """raw_data=<byte\sarray\s10\selements>\)"""

        if sys.hexversion < 0x03000000:
            self.assertRegexpMatches(repr(box), regexp)
        else:
            self.assertRegex(repr(box), regexp)

    def test_uuid_box_xmp(self):
        """Verify uuid repr method for XMP UUID box."""
        jp2file = glymur.data.nemo()
        j = Jp2k(jp2file)
        box = j.box[3]

        # Since the raw_data parameter is a sequence of bytes which could be
        # quite long, don't bother trying to make it conform to eval(repr()).
        regexp = r"""glymur.jp2box.UUIDBox\("""
        regexp += """the_uuid="""
        regexp += """UUID\('be7acfcb-97a9-42e8-9c71-999491e3afac'\),\s"""
        regexp += """raw_data=<byte\sarray\s3122\selements>\)"""

        if sys.hexversion < 0x03000000:
            self.assertRegexpMatches(repr(box), regexp)
        else:
            self.assertRegex(repr(box), regexp)

    def test_contiguous_codestream_box(self):
        """Verify contiguous codestream box repr method."""
        jp2file = glymur.data.nemo()
        jp2 = Jp2k(jp2file)
        box = jp2.box[-1]

        # Difficult to eval(repr()) this, so just match the general pattern.
        regexp = "glymur.jp2box.ContiguousCodeStreamBox"
        regexp += "[(]codestream=<glymur.codestream.Codestream\sobject\s"
        regexp += "at\s0x([a-fA-F0-9]*)>[)]"

        if sys.hexversion < 0x03000000:
            self.assertRegexpMatches(repr(box), regexp)
        else:
            self.assertRegex(repr(box), regexp)
