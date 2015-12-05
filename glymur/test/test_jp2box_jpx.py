# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting JPX box layout.
"""

import ctypes
import os
import struct
import sys
import tempfile
import unittest
import warnings

import lxml.etree as ET

import glymur
from glymur import Jp2k
from glymur.jp2box import DataEntryURLBox, FileTypeBox, JPEG2000SignatureBox
from glymur.jp2box import DataReferenceBox, FragmentListBox, FragmentTableBox
from glymur.jp2box import ColourSpecificationBox

from .fixtures import WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG


@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
class TestJPXWrap(unittest.TestCase):
    """Test suite for wrapping JPX files."""

    def setUp(self):
        self.jpxfile = glymur.data.jpxfile()
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
        </data>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tfile:
            tfile.write(raw_xml)
            tfile.flush()
        self.xmlfile = tfile.name

    def tearDown(self):
        os.unlink(self.xmlfile)

    def test_jpx_ftbl_no_codestream(self):
        """Can have a jpx with no codestream."""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile1:
            with open(self.jp2file, 'rb') as fptr:
                tfile1.write(fptr.read())
            tfile1.flush()
            jp2_1 = Jp2k(tfile1.name)
            jp2h = jp2_1.box[2]

            jp2c = [box for box in jp2_1.box if box.box_id == 'jp2c'][0]

            # coff and clen will be the offset and length input arguments
            # to the fragment list box.  dr_idx is the data reference index.
            coff = []
            clen = []
            dr_idx = []

            coff.append(jp2c.main_header_offset)
            clen.append(jp2c.length - (coff[0] - jp2c.offset))
            dr_idx.append(1)

            # Make the url box for this codestream.
            url1 = DataEntryURLBox(0, [0, 0, 0], 'file://' + tfile1.name)
            url1_name_len = len(url1.url) + 1

            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile2:

                j2k = Jp2k(self.j2kfile)
                jp2_2 = j2k.wrap(tfile2.name)

                jp2c = [box for box in jp2_2.box if box.box_id == 'jp2c'][0]
                coff.append(jp2c.main_header_offset)
                clen.append(jp2c.length - (coff[0] - jp2c.offset))
                dr_idx.append(2)

                # Make the url box for this codestream.
                url2 = DataEntryURLBox(0, [0, 0, 0], 'file://' + tfile2.name)

                boxes = [JPEG2000SignatureBox(),
                         FileTypeBox(brand='jpx ',
                                     compatibility_list=['jpx ',
                                                         'jp2 ', 'jpxb']),
                         jp2h]
                with tempfile.NamedTemporaryFile(suffix='.jpx') as tjpx:
                    for box in boxes:
                        box.write(tjpx)

                    flst = FragmentListBox(coff, clen, dr_idx)
                    ftbl = FragmentTableBox([flst])
                    ftbl.write(tjpx)

                    boxes = [url1, url2]
                    dtbl = DataReferenceBox(data_entry_url_boxes=boxes)
                    dtbl.write(tjpx)
                    tjpx.flush()

                    jpx_no_jp2c = Jp2k(tjpx.name)
                    jpx_boxes = [box.box_id for box in jpx_no_jp2c.box]
                    self.assertEqual(jpx_boxes, ['jP  ', 'ftyp', 'jp2h',
                                                 'ftbl', 'dtbl'])
                    self.assertEqual(jpx_no_jp2c.box[4].DR[0].offset, 141)

                    offset = 141 + 8 + 4 + url1_name_len
                    self.assertEqual(jpx_no_jp2c.box[4].DR[1].offset, offset)

    def test_jp2_with_jpx_box(self):
        """If the brand is jp2, then no jpx boxes are allowed."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]
        boxes = jp2.box

        boxes.append(glymur.jp2box.AssociationBox())

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_jpch_jplh(self):
        """Write a codestream header, compositing layer header box."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # The ftyp box must be modified to jpx.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpxb']

        jpch = glymur.jp2box.CodestreamHeaderBox()
        boxes.append(jpch)
        jplh = glymur.jp2box.CompositingLayerHeaderBox()
        boxes.append(jplh)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            jpx = jp2.wrap(tfile.name, boxes=boxes)

            self.assertEqual(jpx.box[-2].box_id, 'jpch')
            self.assertEqual(jpx.box[-1].box_id, 'jplh')

    def test_cgrp(self):
        """Write a color group box."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # The ftyp box must be modified to jpx.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpxb']

        colr_rgb = ColourSpecificationBox(colorspace=glymur.core.SRGB)
        colr_gr = ColourSpecificationBox(colorspace=glymur.core.GREYSCALE)
        box = [colr_rgb, colr_gr]

        cgrp = glymur.jp2box.ColourGroupBox(box=box)
        boxes.append(cgrp)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            jpx = jp2.wrap(tfile.name, boxes=boxes)

            self.assertEqual(jpx.box[-1].box_id, 'cgrp')
            self.assertEqual(jpx.box[-1].box[0].box_id, 'colr')
            self.assertEqual(jpx.box[-1].box[1].box_id, 'colr')

    def test_label_neg(self):
        """Can't write a label box embedded in any old box."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # The ftyp box must be modified to jpx.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpxb']

        lblb = glymur.jp2box.LabelBox("Just a test")
        box = [lblb]

        cgrp = glymur.jp2box.ColourGroupBox(box=box)
        boxes.append(cgrp)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_cgrp_neg(self):
        """Can't write a cgrp with anything but colr sub boxes"""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # The ftyp box must be modified to jpx.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpxb']

        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        box = [xmlb]

        cgrp = glymur.jp2box.ColourGroupBox(box=box)
        boxes.append(cgrp)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_ftbl(self):
        """Write a fragment table box."""
        # Add a negative test where offset < 0
        # Add a negative test where length < 0
        # Add a negative test where ref > 0 but no data reference box.
        # Add a negative test where more than one flst
        # Add negative test where ftbl contained in a superbox.
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # The ftyp box must be modified to jpx.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpxb']

        offset = [89]
        length = [1132288]
        reference = [0]
        flst = glymur.jp2box.FragmentListBox(offset, length, reference)
        ftbl = glymur.jp2box.FragmentTableBox(box=[flst])
        boxes.append(ftbl)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            jpx = jp2.wrap(tfile.name, boxes=boxes)

            self.assertEqual(jpx.box[1].compatibility_list, ['jp2 ', 'jpxb'])
            self.assertEqual(jpx.box[-1].box_id, 'ftbl')
            self.assertEqual(jpx.box[-1].box[0].box_id, 'flst')

    def test_jpxb_compatibility(self):
        """Wrap JP2 to JPX, state jpxb compatibility"""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # The ftyp box must be modified to jpx with jp2 compatibility.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpxb']

        numbers = (0, 1)
        nlst = glymur.jp2box.NumberListBox(numbers)
        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        asoc = glymur.jp2box.AssociationBox([nlst, xmlb])
        boxes.append(asoc)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            jpx = jp2.wrap(tfile.name, boxes=boxes)

            self.assertEqual(jpx.box[1].compatibility_list, ['jp2 ', 'jpxb'])
            self.assertEqual(jpx.box[-1].box_id, 'asoc')
            self.assertEqual(jpx.box[-1].box[0].box_id, 'nlst')
            self.assertEqual(jpx.box[-1].box[1].box_id, 'xml ')
            self.assertEqual(jpx.box[-1].box[0].associations, numbers)
            self.assertEqual(ET.tostring(jpx.box[-1].box[1].xml.getroot()),
                             b'<data>0</data>')

    def test_association_label_box(self):
        """Wrap JP2 to JPX with asoc, label, and nlst boxes"""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # The ftyp box must be modified to jpx with jp2 compatibility.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list = ['jp2 ', 'jpx ']

        label = 'this is a test'
        lblb = glymur.jp2box.LabelBox(label)
        numbers = (0, 1)
        nlst = glymur.jp2box.NumberListBox(numbers)
        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        asoc = glymur.jp2box.AssociationBox([nlst, xmlb, lblb])
        boxes.append(asoc)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            jpx = jp2.wrap(tfile.name, boxes=boxes)

            self.assertEqual(jpx.box[1].compatibility_list, ['jp2 ', 'jpx '])
            self.assertEqual(jpx.box[-1].box_id, 'asoc')
            self.assertEqual(jpx.box[-1].box[0].box_id, 'nlst')
            self.assertEqual(jpx.box[-1].box[0].associations, numbers)
            self.assertEqual(jpx.box[-1].box[1].box_id, 'xml ')
            self.assertEqual(ET.tostring(jpx.box[-1].box[1].xml.getroot()),
                             b'<data>0</data>')
            self.assertEqual(jpx.box[-1].box[2].box_id, 'lbl ')
            self.assertEqual(jpx.box[-1].box[2].label, label)

    def test_empty_data_reference(self):
        """Empty data reference boxes can be created, but not written."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        boxes[1].brand = 'jpx '

        dref = glymur.jp2box.DataReferenceBox()
        boxes.append(dref)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_deurl_child_of_dtbl(self):
        """Data reference boxes can only contain data entry url boxes."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        ftyp = glymur.jp2box.FileTypeBox()
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                dref = glymur.jp2box.DataReferenceBox([ftyp])
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                dref = glymur.jp2box.DataReferenceBox([ftyp])

        # Try to get around it by appending the ftyp box after creation.
        dref = glymur.jp2box.DataReferenceBox()
        dref.DR.append(ftyp)

        boxes.append(dref)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_only_one_data_reference(self):
        """Data reference boxes cannot be inside a superbox ."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # Have to make the ftyp brand jpx.
        boxes[1].brand = 'jpx '

        flag = 0
        version = (0, 0, 0)
        url = 'file:////usr/local/bin'
        deurl = glymur.jp2box.DataEntryURLBox(flag, version, url)
        dref = glymur.jp2box.DataReferenceBox([deurl])
        boxes.append(dref)
        boxes.append(dref)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_lbl_at_top_level(self):
        """Label boxes can only be inside a asoc box ."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # Have to make the ftyp brand jpx.
        boxes[1].brand = 'jpx '

        lblb = glymur.jp2box.LabelBox('hi there')

        # Put it inside the jp2 header box.
        boxes[2].box.append(lblb)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_data_reference_in_subbox(self):
        """Data reference boxes cannot be inside a superbox ."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # Have to make the ftyp brand jpx.
        boxes[1].brand = 'jpx '

        flag = 0
        version = (0, 0, 0)
        url = 'file:////usr/local/bin'
        deurl = glymur.jp2box.DataEntryURLBox(flag, version, url)
        dref = glymur.jp2box.DataReferenceBox([deurl])

        # Put it inside the jp2 header box.
        boxes[2].box.append(dref)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(IOError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_jp2_to_jpx_sans_jp2_compatibility(self):
        """jp2 wrapped to jpx not including jp2 compatibility is wrong."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]

        # Have to make the ftyp brand jpx.
        boxes[1].brand = 'jpx '
        boxes[1].compatibility_list.append('jp2 ')

        numbers = [0, 1]
        nlst = glymur.jp2box.NumberListBox(numbers)
        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        asoc = glymur.jp2box.AssociationBox([nlst, xmlb])
        boxes.append(asoc)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(RuntimeError):
                jp2.wrap(tfile.name, boxes=boxes)

    def test_jp2_to_jpx_sans_jpx_brand(self):
        """Verify error when jp2 wrapped to jpx does not include jpx brand."""
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[idx] for idx in [0, 1, 2, 4]]
        boxes[1].brand = 'jpx '
        numbers = [0, 1]
        nlst = glymur.jp2box.NumberListBox(numbers)
        the_xml = ET.fromstring('<?xml version="1.0"?><data>0</data>')
        xmlb = glymur.jp2box.XMLBox(xml=the_xml)
        asoc = glymur.jp2box.AssociationBox([nlst, xmlb])
        boxes.append(asoc)

        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with self.assertRaises(RuntimeError):
                jp2.wrap(tfile.name, boxes=boxes)


@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
class TestJPX(unittest.TestCase):
    """Test suite for other JPX boxes."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.jpxfile = glymur.data.jpxfile()

    def tearDown(self):
        pass

    def test_flst_lens_not_the_same(self):
        """A fragment list box items must be the same length."""
        offset = [89]
        length = [1132288]
        reference = [0, 0]
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                flst = glymur.jp2box.FragmentListBox(offset, length, reference)
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                flst = glymur.jp2box.FragmentListBox(offset, length, reference)
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(IOError):
                flst.write(tfile)

    def test_flst_offsets_not_positive(self):
        """A fragment list box offsets must be positive."""
        offset = [0]
        length = [1132288]
        reference = [0]
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                flst = glymur.jp2box.FragmentListBox(offset, length, reference)
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                flst = glymur.jp2box.FragmentListBox(offset, length, reference)
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises((IOError, OSError)):
                flst.write(tfile)

    def test_flst_lengths_not_positive(self):
        """A fragment list box lengths must be positive."""
        offset = [89]
        length = [0]
        reference = [0]
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                flst = glymur.jp2box.FragmentListBox(offset, length, reference)
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                flst = glymur.jp2box.FragmentListBox(offset, length, reference)
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(IOError):
                flst.write(tfile)

    def test_ftbl_boxes_empty(self):
        """A fragment table box must have at least one child box."""
        ftbl = glymur.jp2box.FragmentTableBox()
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(IOError):
                ftbl.write(tfile)

    def test_ftbl_child_not_flst(self):
        """A fragment table box can only contain a fragment list."""
        free = glymur.jp2box.FreeBox()
        ftbl = glymur.jp2box.FragmentTableBox(box=[free])
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(IOError):
                ftbl.write(tfile)

    def test_data_reference_requires_dtbl(self):
        """The existance of data reference box requires a ftbl box as well."""
        flag = 0
        version = (0, 0, 0)
        url1 = 'file:////usr/local/bin'
        url2 = 'http://glymur.readthedocs.org'
        jpx1 = glymur.Jp2k(self.jp2file)
        boxes = jpx1.box
        boxes[1].brand = 'jpx '

        deurl1 = glymur.jp2box.DataEntryURLBox(flag, version, url1)
        deurl2 = glymur.jp2box.DataEntryURLBox(flag, version, url2)
        dref = glymur.jp2box.DataReferenceBox([deurl1, deurl2])
        boxes.append(dref)

        with tempfile.NamedTemporaryFile(suffix='.jpx') as tfile:
            with self.assertRaises(IOError):
                jpx1.wrap(tfile.name, boxes=boxes)

    def test_dtbl(self):
        """Verify that we can interpret Data Reference boxes."""
        # Copy the existing JPX file, add a data reference box onto the end.
        flag = 0
        version = (0, 0, 0)
        url1 = 'file:////usr/local/bin'
        url2 = 'http://glymur.readthedocs.org' + chr(0) * 3
        with tempfile.NamedTemporaryFile(suffix='.jpx') as tfile:
            with open(self.jpxfile, 'rb') as ifile:
                tfile.write(ifile.read())

                deurl1 = glymur.jp2box.DataEntryURLBox(flag, version, url1)
                deurl2 = glymur.jp2box.DataEntryURLBox(flag, version, url2)
                dref = glymur.jp2box.DataReferenceBox([deurl1, deurl2])
                dref.write(tfile)

            tfile.flush()

            jpx = Jp2k(tfile.name)

            self.assertEqual(jpx.box[-1].box_id, 'dtbl')
            self.assertEqual(len(jpx.box[-1].DR), 2)
            self.assertEqual(jpx.box[-1].DR[0].url, url1)
            self.assertEqual(jpx.box[-1].DR[1].url, url2.rstrip('\0'))

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

            jpx = Jp2k(tfile.name)

            self.assertEqual(jpx.box[-1].box_id, 'ftbl')
            self.assertEqual(jpx.box[-1].box[0].box_id, 'flst')
            self.assertEqual(jpx.box[-1].box[0].fragment_offset, (4237,))
            self.assertEqual(jpx.box[-1].box[0].fragment_length, (170246,))
            self.assertEqual(jpx.box[-1].box[0].data_reference, (3,))

    def test_rreq3(self):
        """Verify that we can read a rreq box with mask length 3 bytes"""
        rreq_buffer = ctypes.create_string_buffer(74)
        struct.pack_into('>I4s', rreq_buffer, 0, 74, b'rreq')

        # mask length
        struct.pack_into('>B', rreq_buffer, 8, 3)

        # fuam, dcm.  6 bytes, two sets of 3.
        lst = (255, 224, 0, 0, 31, 252)
        struct.pack_into('>BBBBBB', rreq_buffer, 9, *lst)

        # number of standard features: 11
        struct.pack_into('>H', rreq_buffer, 15, 11)

        standard_flags = [5, 42, 45, 2, 18, 19, 1, 8, 12, 31, 20]
        standard_masks = [8388608, 4194304, 2097152, 1048576, 524288, 262144,
                          131072, 65536, 32768, 16384, 8192]
        for j in range(len(standard_flags)):
            mask = (standard_masks[j] >> 16,
                    standard_masks[j] & 0x0000ffff >> 8,
                    standard_masks[j] & 0x000000ff)
            struct.pack_into('>HBBB', rreq_buffer, 17 + j * 5,
                             standard_flags[j], *mask)

        # num vendor features: 0
        struct.pack_into('>H', rreq_buffer, 72, 0)

        # Ok, done with the box, we can now insert it into the jpx file after
        # the ftyp box.
        with tempfile.NamedTemporaryFile(suffix=".jpx") as ofile:
            with open(self.jpxfile, 'rb') as ifile:
                ofile.write(ifile.read(40))
                ofile.write(rreq_buffer)
                ofile.write(ifile.read())
                ofile.flush()

            jpx = Jp2k(ofile.name)

        self.assertEqual(jpx.box[2].box_id, 'rreq')
        self.assertEqual(type(jpx.box[2]),
                         glymur.jp2box.ReaderRequirementsBox)
        self.assertEqual(jpx.box[2].standard_flag,
                         (5, 42, 45, 2, 18, 19, 1, 8, 12, 31, 20))

    def test_nlst(self):
        """Verify that we can handle a number list box."""
        j = Jp2k(self.jpxfile)
        nlst = j.box[12].box[0].box[0]
        self.assertEqual(nlst.box_id, 'nlst')
        self.assertEqual(type(nlst), glymur.jp2box.NumberListBox)

        # Two associations.
        self.assertEqual(len(nlst.associations), 2)

        # Codestream 0
        self.assertEqual(nlst.associations[0], 1 << 24)

        # Compositing Layer 0
        self.assertEqual(nlst.associations[1], 2 << 24)
