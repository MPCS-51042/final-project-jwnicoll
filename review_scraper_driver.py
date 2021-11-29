import csv
from selenium.webdriver import Firefox

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
    try:
        rows = driver.find_elements_by_class_name('review_table_row')
    except:
        return
    for row in rows:
        try:
            review_name = row.find_element_by_class_name('review_icon') \
                             .get_attribute("class")
            if 'fresh' in review_name:
                score = True
            else:
                score = False
            review = row.find_element_by_class_name('the_review').text.strip()
            reviews_and_scores[review] = score
        except:
            continue

def crawl_reviews(driver, reviews_url, page_count=50):
    '''
        This function processes all critic reviews on the Rotten Tomatoes
        website associated with a single movie. We obtain a dict object
        mapping the text of reviews to whether the review was
        positive or negative.

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
    try:
        driver.get(reviews_url)
        driver.set_page_load_timeout(30)
    except:
        try:
            driver.get(reviews_url)
            driver.set_page_load_timeout(30) 
        except:
            return {}
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

def read_movie_page(movie_url, reviews):
    '''
        This function collects all of the information we want
        for a single movie.

        Inputs:
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
    driver = Firefox()
    driver.implicitly_wait(3)
    try:
        driver.get(movie_url)
        driver.set_page_load_timeout(30)
    except:
        try:
            driver.get(movie_url)
            driver.set_page_load_timeout(30) 
        except:
            driver.quit()
            return
    try:
        scoreboard = driver.find_element_by_class_name('thumbnail-scoreboard-wrap')
        title_tag = scoreboard.find_element_by_tag_name('button')
        title = title_tag.get_attribute('data-title')
        ratings = scoreboard.find_element_by_tag_name('score-board')
        audience_score = ratings.get_attribute('audiencescore')
        tomatometer_score = ratings.get_attribute('tomatometerscore')
        grade = ratings.get_attribute('tomatometerstate')
        revs = driver.find_element_by_class_name('view_all_critic_reviews')
        reviews_url = revs.get_attribute('href')
    except:
        driver.quit()
        return
    reviews[title] = [crawl_reviews(driver, reviews_url), audience_score, \
                      tomatometer_score, grade]
    driver.quit()

def find_urls(all_movies_url, num_clicks):
    '''
        A function to find the urls for movie pages I want to scrape.
        It is more convenient to acquire this list before collecting
        reviews and movie ratings because of how Rotten Tomatoes displays
        its information.

        Inputs:
            all_movies_url: A str object containing the url of the page
                containing all movies with information stored on
                Rotten Tomatoes.

            num_clicks: An int indicating how many times "Show More" should be
                clicked. Each click displays an additional 32 movies.
        
        Returns: A list of urls of Rotten Tomatoes pages to crawl.
    '''
    driver = Firefox()
    driver.get(all_movies_url)
    driver.implicitly_wait(3)
    clicks = 0
    more_movies = driver.find_element_by_class_name('btn-secondary-rt')
    while clicks < num_clicks:
        clicks += 1
        try:
            more_movies.click()
        except:
            break
    movies = driver.find_elements_by_class_name('mb-movie')
    url_list = []
    for movie in movies:
        movie_url = movie.find_element_by_tag_name('a').get_attribute('href')
        url_list.append(movie_url)
    driver.quit()
    return url_list

def find_matches(imdb_titles, url_list):
    '''
        A function to find which urls correspond to movies for which I also
        have data from IMDb, since these are the movies I am interested in.

        Inputs:
            imdb_titles: list object of imdb movie titles.

            url_list: list object of Rotten Tomatoes movie pages to crawl.
        
        Returns:
            A list object containing the urls of Rotten Tomatoes movie pages
                for which I also have IMDb data.
    '''
    urls = []
    for url in url_list:
        try:
            driver = Firefox()
            driver.implicitly_wait(3)
            driver.get(url)
            driver.set_page_load_timeout(30)
            scoreboard = driver.find_element_by_class_name('thumbnail-scoreboard-wrap')
            title_tag = scoreboard.find_element_by_tag_name('button')
            title = title_tag.get_attribute('data-title')
            if title in imdb_titles:
                urls.append(url)
            driver.quit()
        except:
            driver.quit()
            continue
    return urls

def find_reviews(url_list):
    '''
        This function generates the reviews dictionary described above from
        from the Rotten Tomatoes page containing all movies with information
        stored on the site.

        Inputs:
            url_list: A list object containing strings with the urls of movie
                pages that we will scrape.
        
        Returns:
            The reviews dict object described in read_movie_page.
    '''
    reviews = {}
    for url in url_list:
        try:
            read_movie_page(url, reviews)
        except:
            continue
    return reviews

def get_reviews_and_scores(url_list):
    '''
        This function builds the reviews object described above, and generates
        csv files, as described in those functions' doc strings.
        Inputs:
            url_list: list object described in find_reviews
        
        Returns:
            Nothing is returned, but the csv files are generated.
    '''
    reviews = find_reviews(url_list)
    gen_csv(reviews, 'rottentomatoes.csv', sa_scores=False)
    gen_csv_reviews_text(reviews, 'reviewstext.csv')

##################################################################
# Storing Data.
##################################################################
def gen_csv(reviews, file_name, sa_scores):
    '''
        A function which creates a csv file whose rows contain a movie title,
        that movie's audience score, its critic score, and its
        Rotten Tomatoes rating. The sentiment analyzer score is provided
        if sa_scores is True.

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
            The dict object reviews, as described in read_movie_page.
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
