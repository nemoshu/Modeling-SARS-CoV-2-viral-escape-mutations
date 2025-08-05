from utils import *

def load_data():
    """
    Loads fitness data from 'data/fitness_results.txt'.

    Returns:
        pd.DataFrame: Each row contains:
            model (string): Name of the model
            protein (string): Name of the protein
            strain (string): Name of the virus strain
            value (float): Fitness value

    Input File:
        data/fitness_results.txt
    """
    data = []
    fname = 'data/fitness_results.txt'
    with open(fname) as f:
        header = f.readline().rstrip().split('\t')
        proteins = [ field.split(', ')[0] for field in header[1:] ]
        strains = [ field.split(', ')[1] for field in header[1:] ]
        for line in f:
            fields = line.rstrip().split('\t')
            model = fields[0]
            values = [ float(field) for field in fields[1:] ]
            assert(len(proteins) == len(strains) == len(values))
            for prot, strain, value in zip(proteins, strains, values):
                if np.isnan(value):
                    continue
                data.append([ model, prot, strain, value ])
    df = pd.DataFrame(data, columns=[ 'model', 'prot', 'strain', 'value' ])
    return df

def plot_cscs_fitness(df):
    """
    Plots grouped bar chart comparing two neural model scores across strains.

    Args:
        df (pd.DataFrame): Data frame obtained from load_data().

    Returns:
        None

    Output File:
        figures/fitness_barplot_cscs.svg
    """
    df_subset = df[(df['model'] == 'Semantic change') |
                   (df['model'] == 'Grammaticality')]

    plt.figure()
    sns.barplot(data=df_subset, x='strain', y='value', hue='model',
                palette=[ '#ADD8E6', '#FFAAAA' ])
    plt.savefig('figures/fitness_barplot_cscs.svg')

def plot_fitness_benchmark(df):
    """
    Plots fitness benchmark curves for each strain showing performance of multiple models.

    Args:
        df (pd.DataFrame): Data frame obtained from load_data().

    Returns:
        None

    Output File:
        figures/fitness_barplot_benchmark_{strain}.svg
    """
    model_order = [ 'mafft', 'EVcouplings (indep)', 'EVcouplings (epist)',
                    'Grammaticality' ]
    for strain in [ 'WSN33', 'BF520', 'BG505' ]:
        df_subset = df[(df['strain'] == strain) &
                       (df['model'] != 'Semantic change')]
        plt.figure()
        sns.barplot(data=df_subset, x='model', y='value',
                    order=model_order)
        plt.savefig('figures/fitness_barplot_benchmark_{}.svg'
                    .format(strain))

if __name__ == '__main__':
    """
    Plots grouped bar chart comparing two neural model scores across strains,
    and fitness benchmark curves for each strain showing performance of multiple models.
    """
    df = load_data()

    plot_cscs_fitness(df)

    plot_fitness_benchmark(df)
