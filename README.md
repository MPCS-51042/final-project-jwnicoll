# To Run:
    Please download the following from https://drive.google.com/drive/folders/1eBTzxN2jFROtyMJRz_A9nRziIw8pAbln?usp=sharing

        scores_commonurls.csv
        imdb_scores.csv
        sentiment_strengths.csv

    These files will allow you to run all of the code in the jupyter notebook.

    In addition, the following file provides the csv containing reviews from the Rotten Tomatoes web scrape:
    
        reviews_commonurls.csv
    
    The following is the same as scores_commonurls.csv before sentiment analyzer scores were added:
        reviews_scores_commonurls.csv
    
    If that link doesn't work, try this one: https://drive.google.com/drive/folders/1Gmg64lwcC7tWt1qKIgefsgXkpksW1dJk?usp=sharing

# File Summary:

    review_scraper_driver.py: This module performs web scraping on the Rotten Tomatoes website and handles storage of the data.
        Sample urls:
            Dune's reviews: 'https://www.rottentomatoes.com/m/dune_2021/reviews'
            Dune's movie home page: 'https://www.rottentomatoes.com/m/dune_2021'
            Page with all of Rotten Tomatoes' movies: 'https://www.rottentomatoes.com/browse/dvd-streaming-all/'

    imdb_scraper.py: This module performs web scraping on the IMDb website and handles storage of the data.
        Sample url:
            IMDb movies page: 'https://www.imdb.com/search/title/?num_votes=10000,&sort=user_rating,desc&title_type=feature'

    sentimentanalyzer.py: This module builds the sentiment analyzer used to rescore movies based on Rotten Tomatoes critic reviews.

    trainer.py: This module trains a tuning parameter used in building the sentiment analyzer.

    sentiment_analyzer_builder.py: This module expedites the initiation of a sentiment analyzer from stored data.

    rescoring.py: This module uses a sentiment analyzer to rescore movies based on Rotten Tomatoes critic reviews.
    It offers a function to rescore all movies in a csv and save the data if reviews have already been collected.
    It also offers a function to scrape for reviews and score a new movie.

    scores_data_analysis.py: This module handles operations on pandas DataFrames generated from data that has already been collected.
    It offers a function to split reviews into training and test sets.
    It also offers functions to make a presentation of the analysis cleaner.

# Natural Language Toolkit Citations:

Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python: analyzing text with the natural language toolkit. " O&#x27;Reilly Media, Inc."

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# Proposals
Name: Jake Nicoll
# This first idea was approved, and I'm excited for it, so my execution plan pertains to Project idea #1.
Project idea #1: Alternative Rotten Tomatoes Scores
I’m interested in two phenomena I see on Rotten Tomatoes. One is that critics' scores are sometimes different from the audience’s score. The other is that the scores are based on a distribution of binary responses (people either like the movie or they don’t). I think this sometimes leads to scores failing to reflect people’s true opinions of the movie. (An example of this is that every Marvel movie receives a 90% on Rotten Tomatoes. These movies are generally good, but I suspect that score reflects that MOST people think the movie is good and vote accordingly, but that MOST people also don’t believe the movie is a 9/10.)
I think this second phenomena lends itself to a more tractable data science question. I want to see if I can train a sentiment analyzer which will be able to use reviews to generate a score for movies that reflects how good people thought the movies were, rather than what percentage of people liked them (which is a different metric)
I plan to do the following:
Webscrape on Rotten Tomatoes to obtain a sample of reviews associated with movies.
Use NLTK and train a sentiment analyzer (The reviews actually do score movies, so this lends itself to a natural way to train the analyzer).
Re-score movies in the training sample using the sentiment analyzer, and see how these scores compare to the one given by Rotten Tomatoes.
Maybe compare the scores to IMDB scores to gauge how well the new method performs.
Given a webpage for a Rotten Tomatoes movie, scrape the comments and return the sentiment analyzer score.

I know this is a pretty coarse idea, but I’m really interested in doing an NLP project, and I’ve been thinking about my problems with the way Rotten Tomatoes presents scores for a while. I’d be happy to discuss this and see how I could turn this into a good project. This is my top choice.


Project idea #2: Impact of Cook County School Locations on Student Performance
The idea here is to try to figure out how far kids have to travel to a school on average, and how that proximity impacts their performance in school.
I plan to:
Find locations of schools on Chicago city data portal.
Determine the average proximity of students to their schools.
This is tricky. Accurate averages will be hard to calculate, and commute duration is really a better metric of proximity than distance, so I’ll have to think about a reasonable way to do this.
Produce a heat map showing average proximity by school attendance boundary.
The .sh files of attendance boundaries are on the Chicago City Data portal, and I’ve done something like this with .sh files, geopandas, and matplotlib in the past. If I do go this route, I’d be happy to show what I did before (with different .sh files and a different context) to make sure I’m not self-plagiarizing.
For a given attendance boundary, return the average proximity.
Compile metrics of student performance by school from the Chicago City Data portal.
Implement a method that would allow users to select a given performance metric and see the correlation between average proximity and that metric.
The hypothesis would be that longer commutes correspond to worse performance in school.


Project idea #3 (Optional): Don’t have anything in mind right now.

############################################################################################################################################
# To do list and time frames for Project idea 1:

By the end of week 5, I will have explored NLTK so that I know what type of input I need for my sentiment analyzer. (The Rotten Tomatoes reviews do come with scores, so I think this lends itself to a natural way to train the analyzer.) By this point, I will also have started writing my webcrawler for the Rotten Tomatoes website.
    I need to know how to format the information I get from the webcrawler, and I have never used NLTK before, so I need to start with this step before I can do anything else.
    This is also the part of the project I am most worried about. Since I don't know how NLTK works, I am not sure if it can be used to do exactly what I want it to do. If training the sentiment analyzer ends up being convoluted, I want to know that early. I can also look for other open source natural language processing packages at this point if I feel NLTK cannot work for me.

By the end of week 6, I will have a web crawling program that can navigate the Rotten Tomatoes website and collect a sample of reviews associated with movies.
    This will be a large part of the project. However, I want to get this done relatively quickly because training the sentiment analyzer and performing the webscraping go hand in hand. Completing this task now will further inform me about the feasibility of using NLTK to accomplish my goals for this project.
    I will also keep track of the movies, audience scores, and critics' scores for the movies I obtain information for. I want to use this later when I re-score the movies whose reviews I used to train my sentiment analyzer.

By the end of week 7, I will have a trained sentiment analyzer.
    This step will mostly involve just running code that I've written. I am building in some padding here in case the previous two tasks take longer than expected.
    Also, I want to spend a little bit of time here testing my sentiment analyzer to see if it seems reasonably well trained. It could be the case that I do not obtain a sufficient amount of data from my Rotten Tomatoes webcrawling run, in which case I might have to modify the webcrawler or scramble to write another one to scrape reviews from IMDB (or come up with some other solution).

By the end of week 8, I will re-score movies in the training sample using the sentiment analyzer, and see how these scores compare to the ones given by Rotten Tomatoes.
    This is why I kept movies, audience scores, and critics' before. This re-scoring should give me a lot of data to compare sentiment analyzer scores and the original Rotten Tomatoes audience scores. I can also compare to critcs' scores to see how these align with the other two metrics.

By the end of week 9, I will create an additional function that takes a webpage for a Rotten Tomatoes movie, scrapes the comments, and returns the sentiment analyzer score.
    I think the difficulty here is writing another program for webscraping. However, since I am still using the Rotten Tomatoes site, I expect to be able to reuse code from my original crawler.
    This function isn't necessary for comparing sentiment analyzer scores to Rotten Tomatoes audience scores. However, if it turns out that the two scores appear to be qualitatively different, then this function may provide useful information.

If I have time, I might explore finding a way to systematically compare the sentiment analyzer scores, Rotten Tomatoes audience scores, and IMDB scores to gauge whether the sentiment analyzer scores align better with IMDB scores than audience scores.
    These final touches would largely be about refining my analysis to more precisely answer my initial question, which is whether a movie score that reflects how good people thought a movie was is a different metric than the percentage of people that liked the movie (which is the metric that the Rotten Tomatoes audience score provides). At this point, the program will do what I said it will do, and the implementations from week 8 should allow me to provide a reasonable answer to this question. However, it would be cool to take an extra step, time permitting, and see if sentiment analyzer scores better aligned with the IMDB scores than the Rotten Tomatoes audience scores. Since IMDB scores are meant to reflect the quality of a movie, I think such a finding would suggest that the sentiment analyzer scores are in some way better than (and not just different from) the audience scores for assessing the quality of a movie.