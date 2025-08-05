from utils import *

from flair.data import Sentence
from flair.models import SequenceTagger
import nltk
from nltk.corpus import wordnet as wn
from pattern3.en import conjugate, singularize

info_prefix = 'Original headline: '

def extract_content(line):
    """
    Utility to extract information from a line in headline log file.

    Args:
        line (string): Headline log line to parse

    Returns:
        content (string): Headline parsed from the line
    """
    return (': '.join(' | '.join(line.split(' | ')[1:])
                      .strip().split(': ')[:-1]))

def read_headline_info(f, orig_line):
    """
    Builds a dataset from a headline log file.

    Args:
        f (file object): The file to read from
        orig_line (string): The line beginning with the info prefix 'Original headline: '

    Returns:
        info (dict[string, string]: A dictionary containing headline with moderate, least and most change
    """
    orig = orig_line[len(info_prefix):]

    info = { 'orig': orig }

    assert(f.readline().rstrip().endswith('Modifications:'))

    info['mod'] = extract_content(f.readline())
    info['mod_1'] = extract_content(f.readline())
    info['mod_2'] = extract_content(f.readline())

    assert(f.readline().rstrip().endswith('Least change:'))

    info['least_0'] = extract_content(f.readline())
    info['least'] = extract_content(f.readline())
    info['least_1'] = extract_content(f.readline())

    assert(f.readline().rstrip().endswith('Most change:'))

    info['most'] = extract_content(f.readline())
    info['most_1'] = extract_content(f.readline())
    info['most_2'] = extract_content(f.readline())

    return info

def find_diff(headline1, headline2):
    """
    Identifies the first index at which two headlines differ.

    Args:
        headline1 (string): The first tokenized headline
        headline2 (string): The second tokenized headline

    Returns:
        idx (int | None): The index of the first headline, None if no difference found
    """
    for idx, (word1, word2) in enumerate(
            zip(headline1.split(' '), headline2.split(' '))):
        if word1 != word2:
            return idx

flair_pos = None
def pos_tag(sentence, backend='nltk'):
    """
    Generates POS tags for tokens.

    Args:
        sentence (string): raw headline sentence
        backend (string): 'nltk' or 'flair'

    Returns:
        parsed(list[tuple[str, str]]): List of tuples of tokens and their POS tags
    """
    global flair_pos

    if backend == 'nltk':
        return nltk.pos_tag(sentence.split(' '))

    elif backend == 'flair':
        if flair_pos is None:
            flair_pos = SequenceTagger.load('pos')
        sentence_info = Sentence(sentence)
        flair_pos.predict(sentence_info)
        tagged = sentence_info.to_tagged_string().split(' ')
        assert(len(tagged) % 2 == 0)
        parsed = []
        for i in range(len(tagged) // 2):
            idx = i * 2
            tag = tagged[idx + 1]
            assert(tag.startswith('<') and tag.endswith('>'))
            parsed.append((tagged[idx], tag))
        return parsed

    else:
        raise ValueError('Invalid backend: {}'.format(backend))

def find_semantic_distance(wordtag1, wordtag2):
    """
    Evaluates the semantic closeness of edits.

    Args:
        wordtag1 (Tuple[str, str]): The first word and POS pair
        wordtag2 (Tuple[str, str]): The second word and POS pair
    Returns:
        tuple[float, float]: The semantic distance between the two words, in path similarity and Wu-Palmer similarity
        None if the semantic distance can't be computed
    """
    word1, tag1 = wordtag1
    word2, tag2 = wordtag2

    if tag1.startswith('NN') and tag2.startswith('NN'):
        if tag1 == 'NNS':
            word1 = singularize(word1)
        if tag2 == 'NNS':
            word2 = singularize(word2)
        try:
            syn1 = wn.synset('{}.n.01'.format(word1))
            syn2 = wn.synset('{}.n.01'.format(word2))
        except nltk.corpus.reader.wordnet.WordNetError:
            return None
        return syn1.path_similarity(syn2), syn1.wup_similarity(syn2)

    if tag1.startswith('VB') and tag2.startswith('VB'):
        try:
            word1 = conjugate(word1, 'inf')
        except RuntimeError:
            pass
        try:
            word2 = conjugate(word2, 'inf')
        except RuntimeError:
            pass
        try:
            syn1 = wn.synset('{}.v.01'.format(word1))
            syn2 = wn.synset('{}.v.01'.format(word2))
        except nltk.corpus.reader.wordnet.WordNetError:
            return None
        return syn1.path_similarity(syn2), syn1.wup_similarity(syn2)

    return None

def part_of_speech(infos, backend='nltk', n_most=10):
    """
    Performs part of speech analysis and prints out results for:

    - Categories
    - Most common words/POS changes
    - Number of POS changes and statics
    - Semantic similarity measures (if using nltk)
    - t-tests

    Args:
        infos (list[dict]): a compiled list of headline info dictionaries from read_headline_info()
        backend (string): POS tagger backend. Defaults to 'nltk'.
        n_most (int): Number of most common words/POS changes to return. Defaults to 10.

    Returns:
        None
    """
    categories = [ 'mod', 'least', 'most' ]

    # initialize data containers
    pos_diff_change = { category: [] for category in categories }
    n_pos_change = { category: [] for category in categories } # count of POS tag changes
    pct_pos_change = { category: [] for category in categories } # percentage of POS tage changes

    if backend == 'nltk':
        wordnet_path_dist = { category: [] for category in categories }
        wordnet_wup_dist = { category: [] for category in categories }

    for info in infos:
        for category in categories:
            diff_idx = find_diff(info['orig'], info[category]) # first difference

            pos_orig = pos_tag(info['orig'], backend=backend)
            pos_category = pos_tag(info[category], backend=backend)
            assert(len(pos_orig) == len(pos_category)) # lengths should match for comparison

            word_orig = pos_orig[diff_idx]
            word_cat = pos_category[diff_idx]

            if (word_orig[1].startswith('NN') and word_cat[1].startswith('NN')) or \
               (word_orig[1].startswith('VB') and word_cat[1].startswith('VB')):
                pos_diff_change[category].append((word_orig, word_cat))

            n_pos_change[category].append(sum([
                pos_orig[i][1] != pos_category[i][1]
                for i in range(len(pos_orig))
            ]))
            pct_pos_change[category].append(sum([
                pos_orig[i][1] != pos_category[i][1]
                for i in range(len(pos_orig))
            ]) / float(len(pos_orig)) * 100.)

            if backend == 'nltk':
                dists = find_semantic_distance(word_orig, word_cat)
                if dists is not None:
                    path_dist, wup_dist = dists
                    wordnet_path_dist[category].append(path_dist)
                    wordnet_wup_dist[category].append(wup_dist)

    # print out statistics
    for category in categories:
        print('Category: {}'.format(category))

        print('\tMost common word/POS changes, {}:'.format(category))
        for result, count in Counter(
                pos_diff_change[category]).most_common(n_most):
            print('\t{}: {}'.format(result, count))

        change_distr = np.array(n_pos_change[category])
        print('\tNumber of POS changes, mean: {:.4f}, median: {}, '
              'min: {}, max: {}, std: {:.4f}'
              .format(np.mean(change_distr), np.median(change_distr),
                      change_distr.min(), change_distr.max(),
                      change_distr.std()))

        change_distr = np.array(pct_pos_change[category])
        print('\tPercentage of POS changes, mean: {:.4f}%, median: {:.4f}%, '
              'min: {:.4f}%, max: {:.4f}%, std: {:.4f}%'
              .format(np.mean(change_distr), np.median(change_distr),
                      change_distr.min(), change_distr.max(),
                      change_distr.std()))

        if backend == 'nltk':
            sem_distr = np.array(wordnet_path_dist[category])
            print('\tSemantic path similarity of changes ({} total), '
                  'mean: {:.4f}, median: {:.4f}, '
                  'min: {:.4f}, max: {:.4f}, std: {:.4f}'
                  .format(len(sem_distr),
                          np.mean(sem_distr), np.median(sem_distr),
                          sem_distr.min(), sem_distr.max(), sem_distr.std()))
            sem_distr = np.array(wordnet_wup_dist[category])
            print('\tSemantic Wu-Palmer similarity of changes ({} total), '
                  'mean: {:.4f}, median: {:.4f}, '
                  'min: {:.4f}, max: {:.4f}, std: {:.4f}'
                  .format(len(sem_distr),
                          np.mean(sem_distr), np.median(sem_distr),
                          sem_distr.min(), sem_distr.max(), sem_distr.std()))

    print('\tMod vs least POS t-test:')
    print(ss.ttest_ind(np.array(n_pos_change['mod']),
                       np.array(n_pos_change['least'])))
    if backend == 'nltk':
        print('\tMod vs least path-similarity t-test:')
        print(ss.ttest_ind(np.array(wordnet_path_dist['mod']),
                           np.array(wordnet_path_dist['least'])))
        print('\tMod vs least WuP-similarity t-test:')
        print(ss.ttest_ind(np.array(wordnet_wup_dist['mod']),
                           np.array(wordnet_wup_dist['least'])))

def train_topic_model(seqs, vocabulary, n_components=10):
    """
    Trains a topic model.

    Args:
        seqs (dict[tuple[string], list[dict]]): dictionary mapping headline to tokenized word list
        vocabulary (dict[str, int]): a token-index mapping
        n_components (int): number of topics for the LDA model
    Returns:
        LDA model trained on the sequences
    """
    seqs = np.array([ ' '.join(seq) for seq in sorted(seqs.keys()) ])

    X = dok_matrix((len(seqs), len(vocabulary)))
    for seq_idx, seq in enumerate(seqs):
        for word in seq.split(' '):
            X[seq_idx, vocabulary[word] - 1] += 1 # count presence
    X = csr_matrix(X)

    from sklearn.decomposition import LatentDirichletAllocation as LDA
    tprint('LDA, {} components...'.format(n_components))
    model = LDA(n_components=n_components, n_jobs=10).fit(X)

    return model

def lda_topic_model(infos, n_components=10):
    """
    Workflow to create and train LDA topic model:
    - Loads tokenized data
    - Trains LDA model
    - Computes topic assignments and prints the number of headlines which has switched topics

    Args:
        infos (List[Dict]):
        n_components (int): number of topics for the LDA model

    Returns:
        None
    """
    from headlines import setup
    seqs, vocabulary = setup() # load tokenized data

    topic_model = train_topic_model(seqs, vocabulary,
                                    n_components=n_components) # train model

    # compute topic assignments
    X_orig = dok_matrix((len(infos), len(vocabulary)))
    for info_idx, info in enumerate(infos):
        for word in info['orig'].split(' '):
            X_orig[info_idx, vocabulary[word]] += 1
    X_orig_topics = topic_model.transform(X_orig)
    topic_orig = np.argmax(X_orig_topics, axis=1)

    # print statistics
    categories = [ 'mod', 'least', 'most' ]
    for category in categories:
        tprint('Category: {}'.format(category))

        X_category = dok_matrix((len(infos), len(vocabulary)))
        for info_idx, info in enumerate(infos):
            for word in info[category].split(' '):
                X_category[info_idx, vocabulary[word]] += 1
        X_category_topics = topic_model.transform(X_category)
        topic_category = np.argmax(X_category_topics, axis=1)

        tprint('Changed topics: {} / {}'.format(
            sum(topic_orig != topic_category), len(topic_orig)
        ))


if __name__ == '__main__':
    """
    Parses and analyzes headline models, including part-of-speech analysis.
    """
    log_fname = sys.argv[1]

    infos = []
    with open(log_fname) as f:
        for line in f:
            content = line.split(' | ')[-1].strip()
            if content.startswith(info_prefix):
                info = read_headline_info(f, content)
                infos.append(info)

    part_of_speech(infos, backend='nltk')
    part_of_speech(infos, backend='flair')
