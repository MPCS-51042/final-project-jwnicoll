import csv
from selenium.webdriver import Firefox

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
    driver.implicitly_wait(3)
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