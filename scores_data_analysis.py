import pandas as pd

def make_train_test(reviews_text_csv):
    df = pd.read_csv(reviews_text_csv)
    df_train = df.sample(frac=0.5, random_state=0)
    df_test = df.drop(df_train.index)
    df_train.index = range(0, len(df_train))
    df_test.index = range(0, len(df_test))
    return df_train, df_test

# This doesn't need to be in this file.
#def get_revs(df_train):
#    revs = {}
#    for i in range(len(df_train)):
#        rev = df_train['Review'][i]
#        is_pos = df_train['Review is Positive'][i]
#        revs[rev] = is_pos
#    return revs

def rotten_tomatoes_summary_stats(df):
    scores = []
    audience_mean = df['Audience Score'].mean()
    scores.append(('audience mean', audience_mean))
    audience_std = df['Audience Score'].std()
    scores.append(('audience standard deviation', audience_std))
    critic_mean = df['Tomatometer Score'].mean()
    scores.append(('critic mean', critic_mean))
    critic_std = df['Tomatometer Score'].std()
    scores.append(('critic standard deviation', critic_std))
    sa_mean = df['SA Score'].mean()
    scores.append(('sa mean', sa_mean))
    sa_std = df['SA Score'].std()
    scores.append(('sa standard deviation', sa_std))
    vader_mean = df['Vader Compound Score'].mean()
    scores.append(('vader mean', vader_mean))
    vader_std = df['Vader Compound Score'].std()
    scores.append(('vader standard deviation', vader_std))
    df['SA - Critic'] = df['SA Score'] - df['Tomatometer Score']
    SA_crit_diff_mean = df['SA - Critic'].mean()
    scores.append(('sa - crit mean', SA_crit_diff_mean))
    SA_crit_diff_std = df['SA - Critic'].std()
    scores.append(('sa - crit std', SA_crit_diff_std))
    df['SA - Audience'] = df['SA Score'] - df['Audience Score']
    SA_aud_diff_mean = df['SA - Audience'].mean()
    scores.append(('sa - aud mean', SA_aud_diff_mean))
    SA_aud_diff_std = df['SA - Audience'].std()
    scores.append(('sa - aud std', SA_aud_diff_std))
    df['Vader - Critic'] = df['Vader Compound Score'] - df['Tomatometer Score']
    vader_crit_diff_mean = df['Vader - Critic'].mean()
    scores.append(('vader - crit mean', vader_crit_diff_mean))
    vader_crit_diff_std = df['Vader - Critic'].std()
    scores.append(('vader - crit std', vader_crit_diff_std))
    df['Vader - Audience'] = df['Vader Compound Score'] - df['Audience Score']
    vader_aud_diff_mean = df['Vader - Audience'].mean()
    scores.append(('vader - aud mean', vader_aud_diff_mean))
    vader_aud_diff_std = df['Vader - Audience'].std()
    scores.append(('vader - aud std', vader_aud_diff_std))
    return scores

def get_merged_df(rotten_tomatoes_scores_csv, imdb_scores_csv):
    df_rotten_tomatoes = pd.read_csv(rotten_tomatoes_scores_csv)
    df_imdb = pd.read_csv(imdb_scores_csv)
    merged_df = pd.merge(df_rotten_tomatoes, df_imdb, on='Title', how='inner')
    return merged_df

def merged_summary_stats(merged_df):
    scores = []
    imdb_mean = merged_df['IMDb Score'].mean()
    scores.append(('imdb mean', imdb_mean))
    imdb_std = merged_df['IMDb Score'].std()
    scores.append(('imdb std', imdb_std))
    audience_mean = merged_df['Audience Score'].mean()
    scores.append(('audience mean', audience_mean))
    audience_std = merged_df['Audience Score'].std()
    scores.append(('audience standard deviation', audience_std))
    critic_mean = merged_df['Tomatometer Score'].mean()
    scores.append(('critic mean', critic_mean))
    critic_std = merged_df['Tomatometer Score'].std()
    scores.append(('critic standard deviation', critic_std))
    sa_mean = merged_df['SA Score'].mean()
    scores.append(('sa mean', sa_mean))
    sa_std = merged_df['SA Score'].std()
    scores.append(('sa standard deviation', sa_std))
    sa_vs_imdb = merged_df['SA Score'] - merged_df['IMDb Score']
    scores.append(('sa - imdb mean', sa_vs_imdb.mean()))
    scores.append(('sa - imdb std', sa_vs_imdb.std()))
    crit_vs_imdb = merged_df['Tomatometer Score'] - merged_df['IMDb Score']
    scores.append(('crit - imdb mean', crit_vs_imdb.mean()))
    scores.append(('crit - imdb std', crit_vs_imdb.std()))
    #sa_vs_crit = merged_df['SA Score'] - merged_df['Tomatometer Score']
    #scores.append(('sa - crit mean', sa_vs_crit.mean()))
    #scores.append(('sa - crit std', sa_vs_crit.std()))
    #sa_vs_aud = merged_df['SA Score'] - merged_df['Audience Score']
    #scores.append(('sa - aud mean', sa_vs_aud.mean()))
    #scores.append(('sa - aud std', sa_vs_aud.std()))
    return scores
