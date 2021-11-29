import pandas as pd

def make_train_test(reviews_text_csv):
    '''
        A function to produce training and test DataFrames from a csv
        containing movie reviews.

        Inputs:
            reviews_text_csv: A string object containing the name of a
                csv file with one column containing the title of a movie,
                one column containing the text of a review,
                and one column containing a boolean indicating whether the
                review was positive (True) or negative (False).
        
        Returns:
            df_train, df_test: Training and test sets from the csv file.
                Both objects are pandas DataFrame objects with columns as
                in the csv file.
    '''
    df = pd.read_csv(reviews_text_csv)
    df_train = df.sample(frac=0.4, random_state=0)
    df_test = df.drop(df_train.index)
    df_train.index = range(0, len(df_train))
    df_test.index = range(0, len(df_test))
    return df_train, df_test

def get_merged_df(rotten_tomatoes_scores_csv, imdb_scores_csv):
    '''
        A function to produce one DataFrame containing scores for a movie
        from Rotten Tomatoes and IMDb.

        Inputs:
            rotten_tomatoes_scores_csv: A string containing the name of a csv
                file with a column for the title, audience score, tomatometer
                score, rating (rotten, fresh, or certified-fresh), and
                sentiment analyzer score.
            
            imdb_scores_csv: A string containing the name of a csv file with
                a column for the title and a column for the IMDb score.
        
        Returns:
            A pandas DataFrame object with all of the columns of the csv files
                described above.
    '''
    df_rotten_tomatoes = pd.read_csv(rotten_tomatoes_scores_csv)
    df_imdb = pd.read_csv(imdb_scores_csv)
    merged_df = pd.merge(df_rotten_tomatoes, df_imdb, on='Title', how='inner')
    return merged_df

def normalize_imdb_score(score):
    '''
        A function to normalize IMDb scores to the 0-100 scale.

        Inputs:
            score: A float object between 10 and 100, inclusive.

        Returns:
            A float object between 0 and 100, inclusive,
                representing the normalized score.
    '''
    score -= 10
    score *= 10/9
    return score

def add_cols(merged_df):
    '''
        A function to add columns to a DataFrame representing the differences
        between tomatometer and sentiment analyzer scores, sentiment analyzer
        and IMDb scores, and tomatometer and IMDb scores.

        Inputs:
            merged_df: The pandas DataFrame object returned by get_merged_df.
        
        Returns:
            Nothing is returned, but merged_df is modified in place.
    '''
    merged_df['Tomatometer - SA'] = merged_df['Tomatometer Score'] - \
                                    merged_df['SA Score']
    merged_df['SA - IMDb'] = merged_df['SA Score'] - merged_df['IMDb Score']
    merged_df['Tomatometer - IMDb'] = merged_df['Tomatometer Score'] - \
                                      merged_df['IMDb Score']

def compute_stats(merged_df):
    '''
        A function to compute means and standard deviations for the columns
        added to merged_df in add_cols.

        Inputs:
            merged_df: The pandas DataFrame object returned by get_merged_df
                and modified by add_cols.
        
        Returns:
            A tuple containing two list objects. The first list contains the
                means of the columns added by add_cols, and the second list
                contains their standard deviations.
    '''
    tom_minus_sa_mean = merged_df['Tomatometer - SA'].mean()
    tom_minus_sa_std = merged_df['Tomatometer - SA'].std()
    sa_minus_imdb_mean = merged_df['SA - IMDb'].mean()
    sa_minus_imdb_std = merged_df['SA - IMDb'].std()
    tom_minus_imdb_mean = merged_df['Tomatometer - IMDb'].mean()
    tom_minus_imdb_std = merged_df['Tomatometer - IMDb'].std()
    return ([tom_minus_sa_mean, sa_minus_imdb_mean, tom_minus_imdb_mean], \
            [tom_minus_sa_std,  sa_minus_imdb_std, tom_minus_imdb_std])
