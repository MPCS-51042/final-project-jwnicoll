import nltk
import string
import re
import math
import trainer
import scores_data_analysis as sda

# Taken from nltk open source
# https://github.com/nltk/nltk/blob/develop/nltk/sentiment/vader.py#L441
REGEX_REMOVE_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")
PUNC_LIST = [
    ".",
    "!",
    "?",
    ",",
    ";",
    ":",
    "-",
    "'",
    '"',
    "!!",
    "!!!",
    "??",
    "???",
    "?!?",
    "!?!",
    "?!?!",
    "!?!?",
]

PUNCTUATION = string.punctuation
nltk.download(['names', 'stopwords'])
NAMES = nltk.corpus.names.words()
STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.append("n't")

def get_revs(df_train):
    '''
        A function that obtains a mapping of review text to whether the review
        was positive or negative.

        Inputs:
            df_train: A pandas DataFrame object containing a column with movie
                titles, a column with the text of a review for that movie,
                and a column with True (False) indicating the review was
                positive (negative).
        
        Returns:
            A dict object mapping the reviews of movies to numpy.bool_ objects
            indicating whether they are positive (True) or negative (False).
    
    '''
    revs = {}
    for i in range(len(df_train)):
        rev = df_train['Review'][i]
        is_pos = df_train['Review is Positive'][i]
        revs[rev] = is_pos
    return revs

def tokenize(rev):
    '''
        A function to convert the text of a review into a list of words.
        We remove names, punctuation, common words that do not carry sentiment
        and will clutter our analysis, and we force words to be lower case.

        Inputs:
            rev: A str object containing the text of a review

        Returns:
            A list object containing the words in the review.
    '''
    potential_tokens = rev.split()
    tokens = []
    for token in potential_tokens:
        # Remove stopwords and names
        if token in NAMES:
            continue
        token = token.lower()
        token = token.strip(PUNCTUATION)
        if token in STOPWORDS:
            continue
        # Remove singletons and empty strings (after stripping)
        if len(token) <= 1:
            continue
        tokens.append(token)
    return tokens

#CHANGE DOC STRING IF WE GO WITH THIS ALTERNATE IMPLEMENTATION
def create_distributions(revs, n,pos_revs_dist,neg_revs_dist):
    '''
        A function which maps ngrams to the number of times
        they appear in positive and negative reviews.

        Inputs:
            revs: A dict object as described above.

            n: An int object indicating what length ngrams we want.

            pos_revs_dist, neg_revs_dist: dict objects whose keys are ngrams
                and whose values are the number of occurrences of those
                ngrams in positive (negative) reviews.
        
        Returns:
            Nothing is returned. pos_revs_dist and neg_revs_dist
                are modified in place.
    '''
    for rev, is_pos in revs.items():
        tokens = tokenize(rev)
        num_tokens = len(tokens)
        for i in range(num_tokens - n + 1):
            token = (' ').join(tokens[i : i + n])
            if is_pos:
                token_ct = pos_revs_dist.get(token, 0)
                token_ct += 1
                pos_revs_dist[token] = token_ct
            else:
                token_ct = neg_revs_dist.get(token, 0)
                token_ct += 1
                neg_revs_dist[token] = token_ct

def create_big_dist(revs):
    '''
        A function which maps ngrams to the number of times
        they appear in positive and negative reviews.
        We consider ngrams for n = 1, 2, and 3.

        Inputs:
            revs: A dict object as described above.
        
        Returns:
            A tuple containing the pos_revs_dist and neg_revs_dist
                dict objects, as described above..
    '''
    pos_revs_dist = {}
    neg_revs_dist = {}
    for i in range(1,4):
        create_distributions(revs, i, pos_revs_dist, neg_revs_dist)
    #for rev, is_pos in revs.items():
        #tokens = tokenize(rev)
        #num_tokens = len(tokens)
        #for i in range(1, 4):
            #for j in range(num_tokens - i + 1):
            #    token = (' ').join(tokens[j : j + i])
            #    if is_pos:
            #        token_ct = pos_revs_dist.get(token, 0)
            #        token_ct += 1
            #        pos_revs_dist[token] = token_ct
            #    else:
            #        token_ct = neg_revs_dist.get(token, 0)
            #        token_ct += 1
            #        neg_revs_dist[token] = token_ct
    return pos_revs_dist, neg_revs_dist

# Empirically derived alpha:
# 1-grams: 0.784     2-grams: 0.178       3-grams: 0.175
# (^^^ Not trained sequentially, trained discretely)
# ratio: 0.760    zeros: 829
# all-at-once: alpha: 0.204   ratio: 0.793    zeros: 622
# For Reference, vader scored:    ratio: 0.708   zeros: 581
def find_tops(pos_revs_dist, neg_revs_dist, alpha=0.204):
    '''
        This function finds the words which occur most frequently in positive
        and negative reviews. We remove words which are deemed to occur
        frequently in positive and negative reviews, since their sentiment is
        ambiguous.

        Inputs:
            pos_revs_dist, neg_revs_dist: The dict objects described above.

            alpha: A float object indicating the proportion of the smaller of
                pos_revs_dist and neg_revs_dist we wish to use in classifying
                words as frequently occurring. The default value was the
                empirically derived value for this tuning parameter.
            
        Returns:
            Two list objects each with the same number of tuples of words and
                their number of occurrences in positive (negative) reviews.
                Common words are removed. The lists are sorted from most 
                frequently occurring to least.
    '''
    pos_revs_sorted = sorted(pos_revs_dist.items(), \
                            key=lambda x: x[1], reverse=True)
    neg_revs_sorted = sorted(neg_revs_dist.items(), \
                            key=lambda x: x[1], reverse=True)
    k = round(alpha * min(len(pos_revs_sorted), len(neg_revs_sorted)))
    most_common_pos = pos_revs_sorted[0: k]
    most_common_neg = neg_revs_sorted[0: k]
    # Use while loop to remove items
    i = 0
    while i < k:
        j = 0
        pos_item = most_common_pos[i]
        while j < k:
            neg_item = most_common_neg[j]
            if pos_item[0] == neg_item[0]:
                most_common_pos.remove(pos_item)
                most_common_neg.remove(neg_item)
                k -= 1
                i -= 1
                break
            j += 1
        i += 1
    return most_common_pos, most_common_neg

# May want to include num_tiers parameter for finer resolution.
def stratify(most_common_pos, most_common_neg, sentiment_strengths):
    '''
        A function to build sentiment_strengths, which is a dictionary
        whose keys are words and whose values are the sentiment scores
        associated with those words. We divide the most commonly occurring
        words in positive (negative) reviews into four categories, and assign
        scores from 4 to 1 (-4 to -1) based on the categories, with higher
        magnitudes representing stronger sentiment.

        Inputs:
            most_common_pos, most_common_neg: lists of tuples
                as described above.
            
            sentiment_strengths: A dictionary object as described above.
        
        Returns:
            Nothing is returned. sentiment_strengths is modified in place.
    '''
    num_words = len(most_common_pos)
    top = round(num_words / 20)
    quart = round(num_words / 4)
    divs = [0, top, quart, 2 * quart, 3 * quart, num_words]
    num_divs = len(divs)
    for i in range(num_divs - 1):
        for j in range(divs[i], divs[i+1]):
            sentiment_strengths[most_common_pos[j][0]] = (num_divs - 1) - i
            sentiment_strengths[most_common_neg[j][0]] = -(num_divs - 1) + i

# Maybe add flexibility to not have to check for 1, 2, and 3 grams.
def test(df_test, sentiment_strengths):
    '''
        A function which tests sentiment_strengths' ability to classify
        new reviews as positive or negative. If the reviews' sentiment is
        positive (negative), the review is classified as positive (negative).
        Nothing is done for 0 sentiment reviews.

        Inputs:
            df_test: A pandas DataFrame object containing a column with movie
                titles, a column with the text of a review for that movie,
                and a column with True (False) indicating the review was
                positive (negative).
            
            sentiment_strengths: dict object as described previously.
        
        Returns: The proportion of the reviews that were classified correctly,
            of the reviews that were able to be classified.
    '''
    correct = 0
    total = 0
    zeros = 0
    for i in range(len(df_test)):
        rev = tokenize(str(df_test['Review'][i]))
        sentiment = get_sentiment(rev, sentiment_strengths)
        #sentiment = 0
        #num_words = len(rev)
        #for j in range(1, 4):
        #    for k in range(num_words - j + 1):
        #        token = (' ').join(rev[k : k + j])
        #        if j == 1 and k > 0 and rev[k-1] == 'not':
        #            sentiment -= sentiment_strengths.get(token, 0)
        #        else:
        #            sentiment += sentiment_strengths.get(token, 0)
        if sentiment == 0:
            zeros += 1
            continue
        if sentiment > 0:
            is_pos = True
        else:
            is_pos = False
        if is_pos == df_test['Review is Positive'][i]:
            correct += 1
        total += 1
    print(f"zeros: {zeros}")
    return correct / total

def get_sentiment(rev, sentiment_strengths):
    '''
        A function which computes the sentiment strength of a review.
        We add the sentiments of all of the words in the review,
        as determined by sentiment_strengths.
        We then normalize to obtain a score between 0 and 100.

        Inputs:
            rev: A str object containing the text of a review.

            sentiment_strengths: A dict object, as described previously.
    '''
    sentiment = 0
    rev = tokenize(rev)
    num_words = len(rev)
    for j in range(1, 4):
        for k in range(num_words - j + 1):
            token = (' ').join(rev[k : k + j])
            if j == 1 and k > 0 and rev[k-1] == 'not':
                sentiment -= sentiment_strengths.get(token, 0)
            else:
                sentiment += sentiment_strengths.get(token, 0)
    return sentiment

def normalize_score(sentiment):
    # Have to decide what to do about this tuning parameter currently set to 15.
    sentiment = sentiment / math.sqrt((sentiment ** 2) + 15)
    sentiment = (sentiment + 1) * 50
    return sentiment

#######################################################################################
# Own module.
def build_sentiment_strengths(df_train):
    '''
        A function which builds the sentiment strengths dict object as
        described above directly from the training data dataframe.

        Inputs:
            df_train: A pandas DataFrame object as described previously
        
        Returns:
            The sentiment_strengths dict object, as described previously.
    '''
    sentiment_strengths = {}
    revs = get_revs(df_train)
    pos_revs_dist, neg_revs_dist = create_big_dist(revs)
    most_common_pos, most_common_neg = find_tops(pos_revs_dist, neg_revs_dist)
    stratify(most_common_pos, most_common_neg, sentiment_strengths)
    return sentiment_strengths

#def tune_alpha(reviews_text_csv):
#    df_train, df_test = sda.make_train_test(reviews_text_csv)
#    revs = get_revs(df_train)
#    pos_revs_dist, neg_revs_dist = create_big_dist(revs)
#    alpha = trainer.train_alpha(0, 1, 0.1, pos_revs_dist, neg_revs_dist, df_test)[1]
#    return alpha

class SentimentAnalyzer():
    def __init__(self, alpha=None):
        self.sentiment_strengths = {}
        self.alpha = alpha

    def print_alpha(self):
        print(self.alpha)

    # Fix this later to just accept reviews_text.csv
    def train(self, df_train, df_test):
        revs = get_revs(df_train)
        pos_revs_dist, neg_revs_dist = create_big_dist(revs)
        if not self.alpha:
            self.alpha = trainer.train_alpha(0, 1, 0.1, pos_revs_dist, neg_revs_dist, df_test)[1]
        most_common_pos, most_common_neg = find_tops(pos_revs_dist, neg_revs_dist, self.alpha)
        stratify(most_common_pos, most_common_neg, self.sentiment_strengths)

    def get_senti(self, rev):
        return get_sentiment(rev, self.sentiment_strengths)