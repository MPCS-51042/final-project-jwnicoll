import csv
from selenium.webdriver import Firefox
from nltk.sentiment import SentimentIntensityAnalyzer
import sentimentanalyzer as sa
import scores_data_analysis as sda

ALPHA = 0.204

# Dune Reviews URL = 'https://www.rottentomatoes.com/m/dune_2021/reviews'
# Dune Movie Page = 'https://www.rottentomatoes.com/m/dune_2021'
# All Movies Page = 'https://www.rottentomatoes.com/browse/dvd-streaming-all/'
# IMDB = 'https://www.imdb.com/search/title/?num_votes=10000,&sort=user_rating,desc&title_type=feature'       Using this one.
# Or Here: = 'https://www.imdb.com/search/title/?title_type=feature&num_votes=10000,&countries=us&sort=user_rating,desc&ref_=adv_prv'


#########################################################################
# Crawling Rotten Tomatoes
#########################################################################
def read_reviews_page(driver, reviews_and_scores):
    rows = driver.find_elements_by_class_name('review_table_row')
    for row in rows:
        # True for fresh review, False for rotten review.
        try:
            review_name = row.find_element_by_class_name('review_icon').get_attribute("class")
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
    driver.get(reviews_url)
    reviews_and_scores = {}
    # See if there is another next button to click.
    more_reviews = True
    # We won't collect more than 50x20 = 1000 reviews per movie
    # unless otherwise specified.
    count = 0
    while more_reviews and count <= page_count:
        read_reviews_page(driver, reviews_and_scores)
        try:
            driver.find_element_by_class_name('js-prev-next-paging-next').click()
        except:
            more_reviews = False
        count += 1
    return reviews_and_scores

def read_movie_page(driver, movie_url, reviews):
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
    #reviews_url = 'https://www.rottentomatoes.com' + revs['href']
    reviews_url = revs.get_attribute('href')
    reviews[title] = [crawl_reviews(driver, reviews_url), audience_score, tomatometer_score, grade]

def find_reviews(all_movies_url, num_clicks):
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
    #tag = driver.find_element_by_id('main_container')
    #script_tags = tag.find_elements_by_tag_name('script')
    #url_list = []
    #for script in script_tags:
    #    txt = script.get_attribute('innerHTML')
    #    url_list += re.findall('url\":\"[/_\-0-9a-zA-Z]*', txt)
    for url in url_list:
        #movie_url = 'https://www.rottentomatoes.com' + url[6:]
        read_movie_page(driver, url, reviews)
    driver.quit()
    return reviews

def get_reviews_and_scores(all_movies_url, num_clicks):
    reviews = find_reviews(all_movies_url, num_clicks)
    gen_csv(reviews, 'rottentomatoes.csv')
    gen_csv_reviews_text(reviews, 'reviewstext.csv')

def build_sentiment_analyzer(reviews_text_csv):
    df_train, df_test = sda.make_train_test(reviews_text_csv)
    analyzer = sa.SentimentAnalyzer()
    analyzer.train(df_train, df_test)
    return analyzer

#ANALYZER = build_sentiment_analyzer(reviews_text_csv)
###################################################################
# Rescoring Movie
###################################################################
def rescore_movie(movie_url, sentiment_strengths):
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
def add_sentiment_scores(scores_csv, reviews_csv, sentiment_strengths, file_name):
    reviews = gen_revs_from_csvs(scores_csv, reviews_csv)
    for movie, info in reviews.items():
        revs = info[0]
        total_sa_score = 0
        num_revs = 0
        for rev in revs.keys():
            rev = str(rev)
            sa_score = sa.get_sentiment(rev, sentiment_strengths)
            total_sa_score += sa_score
            num_revs += 1
        info[5] = str(total_sa_score / num_revs)
        reviews[movie] = info
    gen_csv_sentiment(reviews, file_name)

##################################################################
# Storing Data.
##################################################################
def gen_csv(reviews, file_name):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        header = ['Title', 'Audience Score', 'Tomatometer Score', 'Rating']
        writer.writerow(header)
        for title, information in reviews.items():
            row = [title] + information[1:]
            writer.writerow(row)

def gen_csv_sentiment(reviews, file_name):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        header = ['Title', 'Audience Score', 'Tomatometer Score', 'Rating', 'SA Score']
        writer.writerow(header)
        for title, information in reviews.items():
            row = [title] + information[1:]
            writer.writerow(row)

def gen_csv_reviews_text(reviews, file_name):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        header = ['Title', 'Review', 'Review is Positive']
        writer.writerow(header)
        for title, information in reviews.items():
            for review_text, grade in information[0].items():
                row = [title] + [review_text] + [grade]
                writer.writerow(row)

# This function might just be for me.
def gen_revs_from_csvs(scores_csv, reviews_csv):
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
                sa_score = line[4]
                info += [audience_score, tomatometer_score, grade, sa_score]
    return reviews

#############################################################################
# Crawling IMDb
#############################################################################
def find_imdb_scores_on_page(driver, imdb_scores):
    movie_names_tags = driver.find_elements_by_class_name('lister-item-header')
    movie_ratings_tags = driver.find_elements_by_class_name('ratings-imdb-rating')
    i = 0
    for name_tag in movie_names_tags:
        title = name_tag.find_element_by_tag_name('a').text
        rating_tag = movie_ratings_tags[i]
        rating = rating_tag.find_element_by_tag_name('strong').text
        # Make sure move doesn't already have a review for some reason.
        if not imdb_scores.get(title):
            imdb_scores[title] = rating
        i += 1

def crawl_imdb_movies(imdb_url):
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
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter = ',')
        header = ['Title', 'IMDb Score']
        writer.writerow(header)
        for title, score in imdb_scores.items():
            score = str(float(score) * 10)
            row = [title] + [score]
            writer.writerow(row)

def imdb_scores_csv(imdb_url, file_name):
    imdb_scores = crawl_imdb_movies(imdb_url)
    gen_csv_imdb_scores(imdb_scores, file_name)