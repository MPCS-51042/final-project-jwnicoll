import csv
from selenium.webdriver import Firefox
from nltk.sentiment import SentimentIntensityAnalyzer
import sentimentanalyzer as sa
import scores_data_analysis as sda

# Dune Reviews URL = 'https://www.rottentomatoes.com/m/dune_2021/reviews'
# Dune Movie Page = 'https://www.rottentomatoes.com/m/dune_2021'
# All Movies Page = 'https://www.rottentomatoes.com/browse/dvd-streaming-all/'
# IMDB = 'https://www.imdb.com/search/title/?num_votes=10000,&sort=user_rating,desc&title_type=feature'       Using this one.
# Or Here: = 'https://www.imdb.com/search/title/?title_type=feature&num_votes=10000,&countries=us&sort=user_rating,desc&ref_=adv_prv'

#########################################################################
# Crawling Rotten Tomatoes
#########################################################################
def read_reviews_page(driver, reviews_and_scores):
    '''
        Collects the reviews on a given page, and maps reviews
        to whether they were positive or negative. Reviews labeled
        'fresh' or 'certified-fresh' are considered positive,
        and reviews labeled 'rotten' are considered negative.

        Inputs:
            driver: A selenium.webdriver Firefox object. The appropriate url
                is passed to the driver before this function is called.

            reviews_and_scores: A dict object mapping the text of a review
                to a boolean indicating whether it is positive (True)
                or negative (False).
        
        Returns:
            Nothing is returned. reviews_and_scores is modified in place.
    '''
    rows = driver.find_elements_by_class_name('review_table_row')
    for row in rows:
        try:
            review_name = row.find_element_by_class_name('review_icon') \
                             .get_attribute("class")
            if 'fresh' in review_name:
                score = True
            else:
                score = False
        except:
            continue
        try:
            review = row.find_element_by_class_name('the_review').text.strip()
            reviews_and_scores[review] = score
        except:
            continue

def crawl_reviews(driver, reviews_url, page_count=50):
    '''
        This function processes all critic reviews on the Rotten Tomatoes
        website associated with a single movie. We obtain a mapping
        of the text of reviews to whether the review was positive or negative.

        Inputs:
            driver: A selenium.webdriver Firefox object.
            
            reviews_url: A str object containing the url of the page with
                all critic reviews for a given movie.
            
            page_count: An int object specifying the maximum number of
                pages to crawl. 20 reviews can be displayed on each page,
                so we will obtain a maximum of 1020 reviews for a given movie
                by default.
            
        Returns:
            The reviews_and_scores dict object which maps the text of each
                review to a boolean indicating whether the review was
                positive or negative.
    '''
    driver.get(reviews_url)
    reviews_and_scores = {}
    # See if there is another next button to click.
    more_reviews = True
    count = 0
    while more_reviews and count <= page_count:
        read_reviews_page(driver, reviews_and_scores)
        try:
            driver.find_element_by_class_name('js-prev-next-paging-next') \
                  .click()
        except:
            more_reviews = False
        count += 1
    return reviews_and_scores

def read_movie_page(driver, movie_url, reviews):
    '''
        This function collects all of the information we want
        for a single movie.

        Inputs:
            driver: A selenium.webdriver Firefox object.

            movie_url: A str object containing the url of the Rotten Tomatoes
                page for a given movie.
            
            reviews: A dict object mapping the title of a movie to a list
                containing the reviews_and_scores dictionary described above,
                the audience score for the movie, the critic score fo the
                movie, and the Rotten Tomatoes grade for the movie.
        
        Returns:
            Nothing is returned by this function. The reviews dictionary
                is modified in place.
    '''
    driver.get(movie_url)
    try:
        scoreboard = driver.find_element_by_class_name('thumbnail-scoreboard-wrap')
    except:
        return
    try:
        title_tag = scoreboard.find_element_by_tag_name('button')
    except:
        return
    title = title_tag.get_attribute('data-title')
    ratings = scoreboard.find_element_by_tag_name('score-board')
    audience_score = ratings.get_attribute('audiencescore')
    tomatometer_score = ratings.get_attribute('tomatometerscore')
    grade = ratings.get_attribute('tomatometerstate')
    revs = driver.find_element_by_class_name('view_all_critic_reviews')
    reviews_url = revs.get_attribute('href')
    reviews[title] = [crawl_reviews(driver, reviews_url), audience_score, \
                      tomatometer_score, grade]

def find_reviews(all_movies_url, num_clicks):
    '''
        This function generates the reviews dictionary described above from
        from the Rotten Tomatoes page containing all movies with information
        stored on the site.

        Inputs:
            all_movies_url: A str object containing the url of the page
                containing all movies with information stored on
                Rotten Tomatoes.
            
            num_clicks: An int indicating how many times "Show More" should be
                clicked. Each click displays an additional 32 movies.
        
        Returns:
            The reviews dict object described above.
    '''
    driver = Firefox()
    driver.get(all_movies_url)
    reviews = {}
    clicks = 0
    more_movies = driver.find_element_by_class_name('btn-secondary-rt')
    while clicks < num_clicks:
        try:
            more_movies.click()
            clicks += 1
        except:
            break
    movies = driver.find_elements_by_class_name('mb-movie')
    url_list = []
    for movie in movies:
        movie_url = movie.find_element_by_tag_name('a').get_attribute('href')
        url_list.append(movie_url)
    for url in url_list:
        read_movie_page(driver, url, reviews)
    driver.quit()
    return reviews

def get_reviews_and_scores(all_movies_url, num_clicks):
    '''
        This function builds the reviews object described above, and generates
        csv files, as described in those functions' doc strings.
        Inputs:
            all_movies_url: A str object, as described above.

            num_clicks: An int as described above.
        
        Returns:
            Nothing is returned, but the csv files are generated.
    '''
    reviews = find_reviews(all_movies_url, num_clicks)
    gen_csv(reviews, 'rottentomatoes.csv', sa_scores=False)
    gen_csv_reviews_text(reviews, 'reviewstext.csv')

###################################################################
# Rescoring Movie
###################################################################
def rescore_movie(movie_url, sentiment_strengths):
    '''
        This function uses the constructed sentiment_strengths dictionary
        to rescore a movie using that movie's reviews.

        Inputs:
            movie_url: A str object containing the Rotten Tomatoes webpage
                of the movie to be rescored.
            
            sentiment_strengths: A dict object mapping words, bigrams, and
                trigrams to values characterizing their association with
                negative and positive reviews.
        
        Returns:
            Nothing is returned. However, this function prints the appropriate
                movie title, audience score, critic score, and the score
                awared by a sentiment analysis of the movie's reviews.
    '''
    driver = Firefox()
    reviews = {}
    read_movie_page(driver, movie_url, reviews)
    driver.quit()
    if reviews:
        title, info = list(reviews.items())[0]
        revs = info[0]
        total_sentiment = 0
        num_reviews = 0
        for rev in revs.keys():
            rev = str(rev)
            sentiment = sa.get_sentiment(rev, sentiment_strengths)
            total_sentiment += sentiment
            num_reviews += 1
        avg_sentiment = total_sentiment / num_reviews
        print((f"Movie Title: {title},\tAudience Score: {info[1]},\t"
               f"Critic Score: {info[2]},\t"
               f"Sentiment Analyzer Score: {avg_sentiment}"))
    else:
        print("No reviews found.")

###################################################################
# Adding Sentiment Scores.
###################################################################
def add_sentiment_scores(scores_csv, reviews_csv, sentiment_strengths, \
                         file_name, reviews=None):
    '''
        This function creates a csv with rows containing a movie title, that
        movie's audience score, its critic score, its Rotten Tomatoes rating,
        and its score awarded by a sentiment analysis of the movie's reviews.

        Inputs:
            scores_csv: A str object containing the name of a csv file
                generated by gen_csv.

            reviews_csv: A str object containing the name of a csv file
                generated by gen_csv_reviews_text.
            
            sentiment_strengths: A dict object as described in rescore movie.

            file_name: A string containing the name of the csv file
                to be created.
            
            reviews: The reviews dict object described above. If it is not
                passed in, we generate it.
        
        Returns:
            Nothing is returned. A csv file as described in gen_csv
                is created.
    '''
    if not reviews:
        reviews = gen_revs_from_csvs(scores_csv, reviews_csv, False)
    for movie, info in reviews.items():
        revs = info[0]
        total_sa_score = 0
        num_revs = 0
        for rev in revs.keys():
            rev = str(rev)
            sa_score = sa.get_sentiment(rev, sentiment_strengths)
            total_sa_score += sa_score
            num_revs += 1
        info += [str(total_sa_score / num_revs)]
        reviews[movie] = info
    gen_csv(reviews, file_name, sa_scores=True)

##################################################################
# Storing Data.
##################################################################
def gen_csv(reviews, file_name, sa_scores):
    '''
        A function which creates a csv file whose rows contain a movie title,
        that movie's audience score, its critic score, and its
        Rotten Tomatoes rating.

        Inputs:
            reviews: A dict object as described in read_movie_page.

            file_name: A str object containing the name of the csv
                file to be made.
            
            sa_scores: A boolean indicating whether sentiment scores
                have been added to the reviews object.

        Returns:
            Nothing is returned, but the csv file described above is created.
    '''
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        header = ['Title', 'Audience Score', 'Tomatometer Score', 'Rating']
        if sa_scores:
            header += ['SA Score']
        writer.writerow(header)
        for title, information in reviews.items():
            row = [title] + information[1:]
            writer.writerow(row)

def gen_csv_reviews_text(reviews, file_name):
    '''
        A function which creates a csv file whose rows contain a movie title,
        the text of a review,
        and True (False) if the review was positive (negative).

        Inputs:
            reviews and file_name as in gen_csv
        
        Returns:
            Nothing is returned, but the appropriate csv file is created.
    '''
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        header = ['Title', 'Review', 'Review is Positive']
        writer.writerow(header)
        for title, information in reviews.items():
            for review_text, grade in information[0].items():
                row = [title] + [review_text] + [grade]
                writer.writerow(row)

def gen_revs_from_csvs(scores_csv, reviews_csv, sa_scores):
    '''
        A function which creates a reviews object from the csv files we
        described above.

        Inputs:
            scores_csv: A str object containing the name of the csv file
                which will have been created using gen_csv.

            reviews_csv: A str object containing the name of the csv file
                created using gen_csv_reviews_text.
            
            sa_scores: A boolean indicating whether sentiment analyzer scores
                are present (True) or absent (False) in the csv.
        
        Returns:
            The dict object reviews, as described above.
    '''
    reviews = {}
    with open(reviews_csv, 'r') as f:
        csv_file = csv.reader(f)
        # Ignore headers.
        next(csv_file)
        for line in csv_file:
            title = line[0]
            review = line[1]
            grade = line[2]
            if grade == 'True':
                grade = True
            else:
                grade = False
            revs = reviews.get(title)
            if revs:
                revs[0][review] = grade
                reviews[title] = revs
            else:
                reviews[title] = [{review: grade}]
    with open(scores_csv, 'r') as f:
        csv_file = csv.reader(f)
        next(csv_file)
        for line in csv_file:
            title = line[0]
            info = reviews.get(title)
            if info:
                audience_score = line[1]
                tomatometer_score = line[2]
                grade = line[3]
                info += [audience_score, tomatometer_score, grade]
                # Add sentiment scores if they are in the csv.
                if sa_scores:
                    sa_score = line[4]
                    info += [sa_score]
    return reviews

#############################################################################
# Crawling IMDb
#############################################################################
def find_imdb_scores_on_page(driver, imdb_scores):
    '''
        A function to associate movie titles to their imdb scores
        on a single page.
        
        Inputs:
            driver: A selenium.webdriver Firefox object, with a url already
                passed in.
            
            imdb_scores: A dict object whose keys are movie titles and
                whose values are strings containing the imdb score fo that
                movie, on a scale of 1 to 10.
        
        Returns:
            Nothing is returned. imdb_scores is modified in place.
    '''
    movie_names_tags = driver.find_elements_by_class_name('lister-item-header')
    ratings_tags = driver.find_elements_by_class_name('ratings-imdb-rating')
    # We subscript because there is no convenient tag for which these
    # are both subtags.
    i = 0
    for name_tag in movie_names_tags:
        title = name_tag.find_element_by_tag_name('a').text
        rating_tag = ratings_tags[i]
        rating = rating_tag.find_element_by_tag_name('strong').text
        # Make sure move doesn't already have a review for some reason.
        if not imdb_scores.get(title):
            imdb_scores[title] = rating
        i += 1

def crawl_imdb_movies(imdb_url):
    '''
        A function to generate the imdb_scores dictionary described above
        for a list of approximately 10000 movies.

        Inputs:
            imdb_url: A str object containing the url for a page with imdb
                movies and scores.

        Returns:
            The imdb_scores dictionary object described above.
    '''
    driver = Firefox()
    driver.get(imdb_url)
    imdb_scores = {}
    # To terminate eventually just in case.
    i = 0
    while i < 10000:
        find_imdb_scores_on_page(driver, imdb_scores)
        try:
            next_tag = driver.find_element_by_class_name('lister-page-next')
            next_url = next_tag.get_attribute('href')
            driver.get(next_url)
        except:
            break
        i += 1
    driver.quit()
    return imdb_scores

def gen_csv_imdb_scores(imdb_scores, file_name):
    '''
        A function to generate a csv file from the imdb_scores object
        described above. The rows of the csv file will contain a movie
        title and its score from 1-10 according to imdb.

        Inputs:
            imdb_scores: A dict object as described above.

            file_name: A str object containing the name of the csv file
                to be created.
        
        Returns:
            Nothing is returned, but the csv file described is created.
    '''
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter = ',')
        header = ['Title', 'IMDb Score']
        writer.writerow(header)
        for title, score in imdb_scores.items():
            score = str(float(score) * 10)
            row = [title] + [score]
            writer.writerow(row)

def imdb_scores_csv(imdb_url, file_name):
    '''
        A function combining the previous two, which creates the desired
        csv file directly from the imdb url.
    '''
    imdb_scores = crawl_imdb_movies(imdb_url)
    gen_csv_imdb_scores(imdb_scores, file_name)