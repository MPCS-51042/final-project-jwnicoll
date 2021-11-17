import sentimentanalyzer as sa

def find_alpha(min_, max_, increment, pos_revs_dist, neg_revs_dist, df_test):
    ratios = []
    i = min_
    while i <= max_:
        if i == 0:
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
    for _ in range(1, 4):
        ratio, alpha = find_alpha(min_, max_, increment, pos_revs_dist, neg_revs_dist, df_test)
        min_ = max(alpha - increment / 2, 0)
        max_ = min(1, alpha + increment /2)
        increment /= 10
        print(f'ratio: {ratio}, alpha: {alpha}\n')
    return ratio, alpha
