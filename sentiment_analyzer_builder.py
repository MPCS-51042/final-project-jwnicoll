import sentimentanalyzer as sa
import csv

def build_sentiment_strengths(df_train):
    '''
        A function which builds the sentiment strengths dict object as
        described above directly from the training data dataframe.

        Inputs:
            df_train: A pandas DataFrame object containing a column with movie
                titles, a column with the text of a review for that movie,
                and a column with True (False) indicating the review was
                positive (negative).
        
        Returns:
            The sentiment_strengths dict object whose keys are ngrams
            and whose values are the sentiment scores associated with
            those ngrams, as described in the sentimentanalyzer.py
            stratify function.
    '''
    sentiment_strengths = {}
    revs = sa.get_revs(df_train)
    pos_revs_dist, neg_revs_dist = sa.create_big_dist(revs)
    most_common_pos, most_common_neg = sa.find_tops(pos_revs_dist, neg_revs_dist)
    sa.stratify(most_common_pos, most_common_neg, sentiment_strengths)
    return sentiment_strengths

def build_sentiment_strengths_123grams(df_train):
    '''
        A function to build sentiment strengths where 1gram, 2gram, and 3gram
        frequency distributions are considered separately. The method above
        performed better on training and test data, but we include this
        function to provide the user flexibility.
        We use empirically derived tuning parameters alpha_1, alpha_2, and
        alpha_3.

        Inputs:
            df_train: As in build_sentiment_strengths.
        
        Returns:
            sentiment_strengths, as in build_sentiment_strengths.
    '''
    sentiment_strengths = {}
    revs = sa.get_revs(df_train)
    one_pos_dist = {}
    one_neg_dist = {}
    two_pos_dist = {}
    two_neg_dist = {}
    three_pos_dist = {}
    three_neg_dist = {}
    sa.create_distributions(revs, 1, one_pos_dist, one_neg_dist)
    sa.create_distributions(revs, 1, two_pos_dist, two_neg_dist)
    sa.create_distributions(revs, 1, three_pos_dist, three_neg_dist)
    one_pos_common, one_neg_common = sa.find_tops(one_pos_dist, \
                                                 one_neg_dist, alpha=sa.ALPHA_1)
    two_pos_common, two_neg_common = sa.find_tops(two_pos_dist, \
                                                 two_neg_dist, alpha=sa.ALPHA_2)
    three_pos_common, three_neg_common = sa.find_tops(three_pos_dist, \
                                                   three_neg_dist, \
                                                   alpha=sa.ALPHA_3)
    sa.stratify(one_pos_common, one_neg_common, sentiment_strengths)
    sa.stratify(two_pos_common, two_neg_common, sentiment_strengths)
    sa.stratify(three_pos_common, three_neg_common, sentiment_strengths)
    return sentiment_strengths

def gen_csv_from_sentiment_strengths(sentiment_strengths, file_name):
    '''
        Generate a csv file from sentiment_strengths for faster reloading.

        Inputs:
            sentiment_strengths: A dict object, as in
                build_sentiment_strengths.

            file_name: A string object containing the name of the
                csv file to be generated.
        
        Returns:
            Nothing is returned, but a csv file with columns containing
                ngrams and their associated sentiment score is generated.
    '''
    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for token, score in sentiment_strengths.items():
            row = [token] + [score]
            writer.writerow(row)

def gen_sentiment_strengths_from_csv(csvfile):
    '''
        Generate sentiment_strengths from a csv file.

        Inputs:
            csvfile: A string containing the name of a csv file generated
                by gen_csv_from_sentiment_strengths.
        
        Returns:
            sentiment_strengths: A dict object, as in
                build_sentiment_strengths.
    '''
    sentiment_strengths = {}
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            sentiment_strengths[str(line[0])] = int(line[1])
    return sentiment_strengths