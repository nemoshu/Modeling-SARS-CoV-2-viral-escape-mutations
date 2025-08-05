from utils import *

import matplotlib

def write_color(chain, resi, acq, of):
    """
    Generates a coloring command for 3D escape potential maps.

    Args:
        chain (string): Protein chain identifier
        resi (string/int): Residue position in the protein
        acq (float): Acquisition score
        of (file object): Output file object for commands

    Returns:
        None
    """
    cmap = matplotlib.cm.get_cmap('viridis')

    of.write('select toColor, resi {} and chain {}\n'
             .format(resi, chain))
    rgb = cmap(acq)
    of.write('color {}, toColor\n'
             .format(matplotlib.colors.rgb2hex(rgb))
             .replace('#', '0x'))

def scale(point, data_min, data_max):
    """
    Normalizes scores between 0 and 1 using min-max scaling.

    Args:
        point (float): Raw acquisition score
        data_min (float): Minimum acquisition score in dataset
        data_max (float): Maximum acquisition score in dataset

    Returns:
        float: Normalized acquisition score
    """
    return (point - data_min) / (data_max - data_min)

def generate_pymol_colors(ofname, df, idx_pdb):
    """
    Generates and writes executable PyMOL script for escape potential maps.

    Args:
        ofname (str): Output file name
        df (pandas.DataFrame): Dataframe containing protein chains, with columns pos and acquisition
        idx_pdb (dict): Maps sequence position to PDB residues

    Returns:
        None
    """
    color_data = []

    # parse the idx_pdb dictionary
    for idx in sorted(set(df['pos'])):
        assert(idx >= 0) # index cannot be negative
        acq_mean = np.mean(df[df['pos'] == idx]['acquisition']) # computes mean acquisition score for each position
        for chain, resi in idx_pdb[idx]:
            color_data.append([ chain, resi, acq_mean ]) # record the parsed color data

    # obtain minimum and maximum for scaling
    acq_min = min([ acq for _, _, acq in color_data ])
    acq_max = max([ acq for _, _, acq in color_data ])

    # write the color data
    with open(ofname, 'w') as of_mean:
        for chain, resi, acq_mean in color_data:
            acq_scaled = scale(acq_mean, acq_min, acq_max)
            write_color(chain, resi, acq_scaled, of_mean)

def load_data(virus, beta=1.):
    """
    Loads data for the specified virus.
    Computes acquisition scores from Equations 3-4 for the purpose of visualisation.

    Args:
        virus (string): Virus identifier
        beta (float): Beta parameter - weight for acquisition function. Defaults to 1.

    Returns:
          A dataframe containing acquisition scores for each chain with the following columns:
            - pos: Residue position
            - prob: Grammatical fitness
            - change: Semantic change
            - acquisition: Acquisition score

    """
    from regional_escape import load
    escape_fname, region_fname = load(virus) # obtain the appropriate file names from regional_escape.py

    # read data to a dataframe
    data = []
    with open(escape_fname) as f:
        columns = f.readline().rstrip().split()
        for line in f:
            data.append(line.rstrip().split('\t'))
    df_all = pd.DataFrame(data, columns=columns)
    df_all['pos'] = pd.to_numeric(df_all['pos'])
    df_all['prob'] = pd.to_numeric(df_all['prob'])
    df_all['change'] = pd.to_numeric(df_all['change'])
    df_all['acquisition'] = ss.rankdata(df_all.change) + \
                            (beta * ss.rankdata(df_all.prob))

    return df_all

def color_lee2019():
    """
    Generates escape potential maps for lee2019 data (H3 Influenza).

    Returns:
        None
    """
    df = load_data('h3')

    # generate index-position map
    idx_pos = {}
    with open('data/influenza/escape_lee2019/avg_sel_tidy.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            idx_pos[int(fields[13])] = fields[4]

    # generate pos_pdb database for graphics
    pos_pdb = {}
    with open('data/influenza/escape_lee2019/'
              'H3_site_to_PDB_4o5n.csv') as f:
        f.readline()
        for line in f:
            pos, chain, resi = line.rstrip().split(',')
            pos_pdb[pos] = [ (chain, resi) ]

    dirname = 'target/influenza/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_h3_mean.pml'

    idx_pdb = { idx: pos_pdb[idx_pos[idx]]
                for idx in sorted(set(df['pos'])) }

    # generate color graph
    generate_pymol_colors(ofname, df, idx_pdb)

def color_doud2018():
    """
    Generates escape potential maps for dod2018 data (H1 Influenza).

    Returns:
        None
    """
    df = load_data('h1')

    idx_pdb = {
        resi: [ ('A', resi), ('B', resi), ('C', resi) ]
        for resi in range(575)
    }

    dirname = 'target/influenza/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_h1_mean.pml'

    generate_pymol_colors(ofname, df, idx_pdb)

def color_dingens2018():
    """
    Generates escape potential maps for dingens2018 data (HIV).

    Returns:
        None
    """
    df = load_data('hiv')
    idxs = sorted(set(df['pos'])) # available indexes

    idx_pdb = {}
    for idx in idxs:
        if idx < 320:
            pos = str(idx + 2)
            idx_pdb[idx] = [ ('G', pos) ]
        elif idx == 320:
            pos = '321A'
            idx_pdb[idx] = [ ('G', pos) ]
        elif idx < 514:
            pos = str(idx + 1)
            idx_pdb[idx] = [ ('G', pos) ]
        else:
            pos = str(idx + 4)
            idx_pdb[idx] = [ ('B', pos) ]

    dirname = 'target/hiv/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_gp120_mean.pml'

    generate_pymol_colors(ofname, df, idx_pdb)

def color_starr2020():
    """
    Generates escape potential maps for starr2020 data (SARS-CoV-2).

    Returns:
        None
    """
    df = load_data('sarscov2')

    idx_pdb = { idx: [ ('A', str(idx + 1)),
                       ('B', str(idx + 1)),
                       ('C', str(idx + 1)) ]
                for idx in sorted(set(df['pos'])) }

    dirname = 'target/cov/structure'
    mkdir_p(dirname)
    ofname = dirname + '/pdb_color_sarscov2_mean.pml'

    generate_pymol_colors(ofname, df, idx_pdb)

if __name__ == '__main__':
    """
    Generates escape potential maps for the four datasets.
    
    Input Files: 
        - as specified in regional_escape.load()
        - data/influenza/escape_lee2019/avg_sel_tidy.csv
        - data/influenza/escape_lee2019/H3_site_to_PDB_4o5n.csv
    
    Output Files:
        - target/influenza/structure/pdb_color_h1_mean.pml (doud2018)
        - target/influenza/structure/pdb_color_h3_mean.pml (lee2019)
        - target/hiv/structure/pdb_color_gp120_mean.pml (dingens2018)
        - target/cov/structure/pdb_color_sarscov2_mean.pml (starr2020)
    """
    color_doud2018()
    color_lee2019()
    color_dingens2018()
    color_starr2020()
