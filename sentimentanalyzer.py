import nltk
import string
import math
import trainer

# https://github.com/nltk/nltk/blob/develop/nltk/sentiment/vader.py#L441

PUNCTUATION = string.punctuation
nltk.download(['names', 'stopwords'])
NAMES = nltk.corpus.names.words()
STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.append("n't")
ALPHA = 0.215
ALPHA_1 = 0.784
ALPHA_2 = 0.178
ALPHA_3 = 0.175

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
            rev: A str object containing the text of a review.

        Returns:
            A list object containing the words in the review.
    '''
    potential_tokens = str(rev).split()
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

def create_distributions(revs, n, pos_revs_dist, neg_revs_dist):
    '''
        A function which maps ngrams to the number of times
        they appear in positive and negative reviews.

        Inputs:
            revs: A dict object as described in get_revs

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
            revs: A dict object as described in get_revs.
        
        Returns:
            A tuple containing the pos_revs_dist and neg_revs_dist
                dict objects, as described in create_distributions.
    '''
    pos_revs_dist = {}
    neg_revs_dist = {}
    for i in range(1,4):
        create_distributions(revs, i, pos_revs_dist, neg_revs_dist)
    return pos_revs_dist, neg_revs_dist

def find_tops(pos_revs_dist, neg_revs_dist, alpha=ALPHA):
    '''
        This function finds the words which occur most frequently in positive
        and negative reviews. We remove words which are deemed to occur
        frequently in positive and negative reviews, since their sentiment is
        ambiguous.

        Inputs:
            pos_revs_dist, neg_revs_dist: The dict objects described in
                create_distributions.

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

def stratify(most_common_pos, most_common_neg, sentiment_strengths):
    '''
        A function to build sentiment_strengths, which is a dictionary
        whose keys are ngrams and whose values are the sentiment scores
        associated with those ngrams. We divide the most commonly occurring
        words in positive (negative) reviews into four categories, and assign
        scores from 5 to 1 (-5 to -1) based on the categories, with higher
        magnitudes representing stronger sentiment.

        Inputs:
            most_common_pos, most_common_neg: lists of tuples
                as described in find_tops.
            
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

def get_sentiment(rev, sentiment_strengths):
    '''
        A function which computes the sentiment strength of a review.
        We add the sentiments of all of the words in the review,
        as determined by sentiment_strengths.

        Inputs:
            rev: A str object containing the text of a review.

            sentiment_strengths: A dict object, as described in the
                stratify function.
        
        Returns:
            An int object representing the sentiment contained in a review.
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
            
            sentiment_strengths: dict object as described in the
                stratify function.
        
        Returns: The proportion of the reviews that were classified correctly,
            of the reviews that were able to be classified.
    '''
    correct = 0
    total = 0
    for i in range(len(df_test)):
        rev = tokenize(str(df_test['Review'][i]))
        sentiment = get_sentiment(rev, sentiment_strengths)
        if sentiment == 0:
            continue
        if sentiment > 0:
            is_pos = True
        else:
            is_pos = False
        if is_pos == df_test['Review is Positive'][i]:
            correct += 1
        total += 1
    return correct / total

def normalize_score(sentiment):
    '''
        A function to normalize a sentiment score to be between 0 and 100.
        The 15 represents a tuning parameter empirically derived in nltk's
        open source file, which we use here without rederiving.

        Inputs:
            sentiment: int object representing sentiment.
        
        Returns:
            A float object representing the  normalized sentiment score.
    '''
    sentiment = sentiment / math.sqrt((sentiment ** 2) + 15)
    sentiment = (sentiment + 1) * 50
    return sentiment
