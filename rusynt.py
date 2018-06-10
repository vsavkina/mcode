from pathlib import Path, PurePosixPath
from rusyntax import *
from conllu.parser import parse, parse_tree

def parse_sent(inf,outf, return_tree = True):

    # read configs and command line options
    config = configparser.ConfigParser()
    config.read('config.ini')
    in_fname, out_fname = inf, outf
    check_infile(in_fname)

    fname_clean = os.path.basename(in_fname).rsplit('.', 1)[0]

    # temporary files and folder
    tmp_path = get_path_from_config(config, 'TMP_PATH', 'tmp')
    tmp_fsuffixes = ['_mystem_in.txt', '_mystem_out.txt',
                     '_treetagger_in.txt', '_treetagger_out.txt',
                     '_raw.conll']
    a,b,c,d,e = ( PurePosixPath(j) for j in tmp_path.split('/'))
    tmp_fnames = [str(a / b / c / d / e / (fname_clean + fsuffix))
                    for fsuffix in tmp_fsuffixes]

    # output file and folder
    out_path = get_path_from_config(config, 'OUT_PATH', 'out')
    a,b,c,d,e = ( PurePosixPath(j) for j in out_path.split('/'))
    if out_fname is None:
        out_fname = str(a / b / c / d / e / (fname_clean + '.conll'))
    else:
        out_fname = str(a / b / c / d / e / out_fname)

    # create output and temp folder if needed
    for path in [tmp_path, out_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # rock'n'roll
    process(in_fname, out_fname,
            config['DEFAULT']['APP_ROOT'],
            config['mystem']['MYSTEM_PATH'],
            config['malt']['MALT_ROOT'],
            config['malt']['MALT_NAME'],
            config['malt']['MODEL_NAME'],
            config['dicts']['COMP_DICT_PATH'],
            config['treetagger']['TREETAGGER_BIN'],
            config['treetagger']['TREETAGGER_PAR'],
            *tmp_fnames)
    
    for fname in tmp_fnames:
        os.remove(fname)

    with open(out_fname, 'r', encoding='utf-8') as conll_file:
        conll_data = conll_file.read()
        conll_file.close()
        os.remove(out_fname)
    if return_tree:
        return parse_tree(conll_data)
    return parse(conll_data)
