decompression_parameters_type = """<class 'glymur.lib.openjp2.DecompressionParametersType'>:
    cp_reduce: 0
    cp_layer: 0
    infile: b''
    outfile: b''
    decod_format: -1
    cod_format: -1
    DA_x0: 0
    DA_x1: 0
    DA_y0: 0
    DA_y1: 0
    m_verbose: 0
    tile_index: 0
    nb_tile_to_decode: 0
    jpwl_correct: 0
    jpwl_exp_comps: 0
    jpwl_max_tiles: 0
    flags: 0"""

default_progression_order_changes_type = """<class 'glymur.lib.openjp2.PocType'>:
    resno0: 0
    compno0: 0
    layno1: 0
    resno1: 0
    compno1: 0
    layno0: 0
    precno0: 0
    precno1: 0
    prg1: 0
    prg: 0
    progorder: b''
    tile: 0
    tx0: 0
    tx1: 0
    ty0: 0
    ty1: 0
    layS: 0
    resS: 0
    compS: 0
    prcS: 0
    layE: 0
    resE: 0
    compE: 0
    prcE: 0
    txS: 0
    txE: 0
    tyS: 0
    tyE: 0
    dx: 0
    dy: 0
    lay_t: 0
    res_t: 0
    comp_t: 0
    prec_t: 0
    tx0_t: 0
    ty0_t: 0"""

default_compression_parameters_type = """<class 'glymur.lib.openjp2.CompressionParametersType'>:
    tile_size_on: 0
    cp_tx0: 0
    cp_ty0: 0
    cp_tdx: 0
    cp_tdy: 0
    cp_disto_alloc: 0
    cp_fixed_alloc: 0
    cp_fixed_quality: 0
    cp_matrice: None
    cp_comment: None
    csty: 0
    prog_order: 0
    numpocs: 0
    numpocs: 0
    tcp_numlayers: 0
    tcp_rates: []
    tcp_distoratio: []
    numresolution: 6
    cblockw_init: 64
    cblockh_init: 64
    mode: 0
    irreversible: 0
    roi_compno: -1
    roi_shift: 0
    res_spec: 0
    prch_init: []
    prcw_init: []
    infile: b''
    outfile: b''
    index_on: 0
    index: b''
    image_offset_x0: 0
    image_offset_y0: 0
    subsampling_dx: 1
    subsampling_dy: 1
    decod_format: -1
    cod_format: -1
    jpwl_epc_on: 0
    jpwl_hprot_mh: 0
    jpwl_hprot_tph_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_hprot_tph: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot_packno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_sens_size: 0
    jpwl_sens_addr: 0
    jpwl_sens_range: 0
    jpwl_sens_mh: 0
    jpwl_sens_tph_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_sens_tph: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cp_cinema: 0
    max_comp_size: 0
    cp_rsiz: 0
    tp_on: 0
    tp_flag: 0
    tcp_mct: 0
    jpip_on: 0
    mct_data: None
    max_cs_size: 0
    rsiz: 0"""

default_image_component_parameters = """<class 'glymur.lib.openjp2.ImageComptParmType'>:
    dx: 0
    dy: 0
    w: 0
    h: 0
    x0: 0
    y0: 0
    prec: 0
    bpp: 0
    sgnd: 0"""

# The "icc_profile_buf" field is problematic as it is a pointer value, i.e.
#
#     icc_profile_buf: <glymur.lib.openjp2.LP_c_ubyte object at 0x7f28cd5d5d90>
#
# Have to treat it as a regular expression.
default_image_type = """<class 'glymur.lib.openjp2.ImageType'>:
    x0: 0
    y0: 0
    x1: 0
    y1: 0
    numcomps: 0
    color_space: 0
    icc_profile_buf: <glymur.lib.openjp2.LP_c_ubyte object at 0x[0-9A-Fa-f]*>
    icc_profile_len: 0"""
