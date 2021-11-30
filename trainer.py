import sentimentanalyzer as sa

def find_alpha(min_, max_, increment, pos_revs_dist, neg_revs_dist, df_test):
    '''
        Function to find the optimal proportion (alpha) of the frequency
        distributions to use between min_ and max_. Our condition for
        optimality is achieving lowest error rate in classifications of
        reviews from df_test in which a classification was achieved
        (sentiment score not equal to 0).

        Inputs:
            min_, max_: float objects representing the minimum and
                maximum alpha to be tested, respectively.

            increment: A float object specifying the amount by which we will
                increase alpha between each test.
            
            pos_revs_dist, neg_revs_dist: dict objects containing frequency
                distributions of words in positive and negative reviews,
                respectively.
            
            df_test: A pandas DataFrame object containing a column with movie
                titles, a column with the text of a review for that movie,
                and a column with True (False) indicating the review was
                positive (negative).
        
        Returns:
            A tuple of float objects whose first element is the highest
                classification success rate achieved across all values
                of alpha tested, and whose second element is the value of
                alpha for which this maximum was achieved.
    '''
    ratios = []
    i = min_
    while i <= max_:
        if i == 0:
            # Set ratio to -1 because we don't want to use alpha=0
            ratios.append((-1, i))
            i += increment
            continue
        most_common_pos, most_common_neg = sa.find_tops(pos_revs_dist, \
                                                        neg_revs_dist, alpha=i)
        sentiment_strengths = {}
        sa.stratify(most_common_pos, most_common_neg, sentiment_strengths)
        ratio = sa.test(df_test, sentiment_strengths)
        ratios.append((ratio, i))
        i += increment
    return max(ratios)

def train_alpha(min_, max_, increment, pos_revs_dist, neg_revs_dist, df_test):
    '''
        A function to iteratively apply find_alpha in order to obtain a precise
        estimate of the optimal value of alpha to use.
        We choose to call this function with min_ = 0.0, max_ = 1.0,
        and increment = 0.1 in order to obtain an estimate of alpha to the
        third decimal place.
        
        Inputs:
            The inputs are the same as in find_alpha.
        
        Returns:
            A tuple of floats, as in find_alpha.
    '''
    for _ in range(1, 4):
        ratio, alpha = find_alpha(min_, max_, increment, \
                                  pos_revs_dist, neg_revs_dist, df_test)
        min_ = max(alpha - increment / 2, 0)
        max_ = min(1, alpha + increment /2)
        increment /= 10
    return ratio, alpha
