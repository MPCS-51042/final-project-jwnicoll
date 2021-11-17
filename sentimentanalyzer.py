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
    revs = {}
    for i in range(len(df_train)):
        rev = df_train['Review'][i]
        is_pos = df_train['Review is Positive'][i]
        revs[rev] = is_pos
    return revs

def tokenize(rev):
    potential_tokens = rev.split()
    tokens = []
    for token in potential_tokens:
    # tokens = rev.split()
    # for token in tokens:
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

# HAVE TO REMOVE STARTING AND ENDING PUCTUATION
def create_distributions(revs, n):
    '''
        Inputs:
            revs (dict): Dictionary mapping reviews as strings to
                booleans indicating whether the review is positive (True)
                or negative.
            n (int): Integer indicating what n grams we want.
    '''
    pos_revs_dist = {}
    neg_revs_dist = {}
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
    return pos_revs_dist, neg_revs_dist

# Empirically derived alpha:
# 1-grams: 0.784     2-grams: 0.178       3-grams: 0.175
# (^^^ Not trained sequentially, trained discretely)
# ratio: 0.760    zeros: 829
# all-at-once: alpha: 0.204   ratio: 0.793    zeros: 622
# For Reference, vader scored:    ratio: 0.708   zeros: 581
def find_tops(pos_revs_dist, neg_revs_dist, alpha=0.204):
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
        Inputs:
            most_common_pos, most_common_neg (list): Lists of tuples containing
                most commonly appearing words in positive (negative) reviews
                and the number of appearances of those words.
                We use the fact that these lists will have the same length.
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
    correct = 0
    total = 0
    zeros = 0
    for i in range(len(df_test)):
        rev = tokenize(str(df_test['Review'][i]))
        sentiment = 0
        num_words = len(rev)
        for j in range(1, 4):
            for k in range(num_words - j + 1):
                token = (' ').join(rev[k : k + j])
                if j == 1 and k > 0 and rev[k-1] == 'not':
                    sentiment -= sentiment_strengths.get(token, 0)
                else:
                    sentiment += sentiment_strengths.get(token, 0)
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

def create_big_dist(revs):
    pos_revs_dist = {}
    neg_revs_dist = {}
    for rev, is_pos in revs.items():
        tokens = tokenize(rev)
        num_tokens = len(tokens)
        for i in range(1, 4):
            for j in range(num_tokens - i + 1):
                token = (' ').join(tokens[j : j + i])
                if is_pos:
                    token_ct = pos_revs_dist.get(token, 0)
                    token_ct += 1
                    pos_revs_dist[token] = token_ct
                else:
                    token_ct = neg_revs_dist.get(token, 0)
                    token_ct += 1
                    neg_revs_dist[token] = token_ct
    return pos_revs_dist, neg_revs_dist

def get_sentiment(rev, sentiment_strengths):
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
    # Have to decide what to do about this tuning parameter currently set to 15.
    sentiment = sentiment / math.sqrt((sentiment ** 2) + 15)
    sentiment = (sentiment + 1) * 50
    return sentiment

def build_sentiment_strengths(df_train):
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