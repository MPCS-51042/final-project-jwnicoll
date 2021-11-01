import bs4
import requests
import csv
import re

# Dune Reviews URL = 'https://www.rottentomatoes.com/m/dune_2021/reviews'
# Dune Movie Page = 'https://www.rottentomatoes.com/m/dune_2021'
# All Movies Page = 'https://www.rottentomatoes.com/browse/dvd-streaming-all/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}


def read_reviews_page(soup, reviews_and_scores):
    rows = soup.find_all('div', class_='row review_table_row')
    for row in rows:
        # True for fresh review, False for rotten review.
        if row.find('div', class_='review_icon icon small fresh'):
            score = True
        else:
            score = False
        review = row.find('div', class_='the_review').text.strip()
        reviews_and_scores[review] = score    

def crawl_reviews(reviews_url):
    source = requests.get(reviews_url, headers=headers).text
    soup = bs4.BeautifulSoup(source, 'html5lib')
    #rows = soup.find_all('div', class_="row review_table_row")
    reviews_and_scores = {}
    read_reviews_page(soup, reviews_and_scores)
    return reviews_and_scores

def read_movie_page(movie_url, reviews):
    source = requests.get(movie_url, headers=headers).text
    soup = bs4.BeautifulSoup(source, 'html5lib')
    scoreboard = soup.find('div', class_='thumbnail-scoreboard-wrap')
    if not scoreboard:
        return
    title_tag = scoreboard.find('button')
    if not title_tag:
        return
    title = title_tag['data-title']
    ratings = scoreboard.find('score-board')
    audience_score = ratings['audiencescore']
    tomatometer_score = ratings['tomatometerscore']
    grade = ratings['tomatometerstate']
    revs = soup.find('a', class_='view_all_critic_reviews')
    reviews_url = 'https://www.rottentomatoes.com' + revs['href']
    reviews[title] = [crawl_reviews(reviews_url), audience_score, tomatometer_score, grade]

def find_reviews(all_movies_url):
    source = requests.get(all_movies_url, headers=headers).text
    soup = bs4.BeautifulSoup(source, 'html5lib')
    reviews = {}
    tag = soup.find('div', id='main_container')
    script_tag = tag.find_all('script')[5]
    script = str(script_tag.contents[0])
    url_list = re.findall('url\":\"[/_\-0-9a-zA-Z]*', script)
    for url in url_list:
        movie_url = 'https://www.rottentomatoes.com' + url[6:]
        read_movie_page(movie_url, reviews)
    return reviews

def gen_csv(reviews, file_name):
    import csv
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        header = ['Title', 'Audience Score', 'Tomatometer Score', 'Rating']
        writer.writerow(header)
        for title, information in reviews.items():
            row = [title] + information[1:]
            writer.writerow(row)



    # Having a hard time pulling this.
    lst = [{"id":771494725,"title":"Needle in a Timestack","url":"/m/needle_in_a_timestack","tomatoIcon":"rotten","tomatoScore":34,"theaterReleaseDate":"Oct 15","dvdReleaseDate":"Oct 15","mpaaRating":"R","synopsis":"Despite a nifty premise and some talented stars, <em>Needle in a Timestack</em> is too slack and diffuse to recommend searching for.","synopsisType":"consensus","runtime":"1 hr. 51 min.","posters":{"thumborId":"v1.bTsxMzkwNjQyMjtqOzE5MDE1OzEyMDA7ODAyNzsxMTkyNw","primary":"https://resizing.flixster.com/zHfTv8GheJ31qRxhihqhz4EQmKg=/130x0/v1.bTsxMzkwNjQyMjtqOzE5MDE1OzEyMDA7ODAyNzsxMTkyNw"},"actors":["Orlando Bloom", "Leslie Odom Jr.", "Cynthia Erivo"]},{"id":771504776,"title":"Horror Noire: A History of Black Horror","url":"/m/horror_noire_a_history_of_black_horror","tomatoIcon":"fresh","tomatoScore":100,"theaterReleaseDate":"Feb 1","dvdReleaseDate":"Oct 28","mpaaRating":"NR","synopsis":"<em>Horror Noire: A History of Black Horror</em> more than lives up to its title, offering a smart and entertaining overview of American film history through an overlooked lens.","synopsisType":"consensus","runtime":"1 hr. 30 min.","posters":{"thumborId":"v1.bTsxMjk4MjE1MjtqOzE5MDA1OzEyMDA7MjAwOzMwMA","primary":"https://resizing.flixster.com/-GwkNC-LPM0x7CeoHaSP4_f9LbM=/130x0/v1.bTsxMjk4MjE1MjtqOzE5MDA1OzEyMDA7MjAwOzMwMA"},"actors":["Ashlee Blackwell"]},{"id":771505502,"title":"Dune","url":"/m/dune_2021","tomatoIcon":"certified_fresh","tomatoScore":83,"theaterReleaseDate":"Oct 22","dvdReleaseDate":"Oct 22","mpaaRating":"NR","synopsis":"<em>Dune</em> occasionally struggles with its unwieldy source material, but those issues are largely overshadowed by the scope and ambition of this visually thrilling adaptation.","synopsisType":"consensus","posters":{"thumborId":"v1.bTsxMzg5NDE3MztqOzE5MDE1OzEyMDA7Mjc2NDs0MDk2","primary":"https://resizing.flixster.com/q-6ibncXKpEf9oWwcSED6X5H31w=/130x0/v1.bTsxMzg5NDE3MztqOzE5MDE1OzEyMDA7Mjc2NDs0MDk2"},"actors":["Timothée Chalamet", "Rebecca Ferguson", "Josh Brolin"]},{"id":771511088,"title":"Halloween Kills","url":"/m/halloween_kills","tomatoIcon":"rotten","tomatoScore":39,"theaterReleaseDate":"Oct 15","dvdReleaseDate":"Oct 15","mpaaRating":"NR","synopsis":"<em>Halloween Kills</em> should satisfy fans in search of brute slasher thrills, but in terms of advancing the franchise, it's a bit less than the sum of its bloody parts.","synopsisType":"consensus","posters":{"thumborId":"v1.bTsxMzk1NTgyOTtqOzE5MDE2OzEyMDA7MzE1ODs1MDAw","primary":"https://resizing.flixster.com/Mu0KG9pkaGTz6o6bzJfqr95wnD8=/130x0/v1.bTsxMzk1NTgyOTtqOzE5MDE2OzEyMDA7MzE1ODs1MDAw"},"actors":["Jamie Lee Curtis", "Anthony Michael Hall", "Judy Greer"]},{"id":771512423,"title":"Held for Ransom (Ser du månen, Daniel)","url":"/m/held_for_ransom","tomatoIcon":"fresh","tomatoScore":100,"theaterReleaseDate":"Oct 15","dvdReleaseDate":"Oct 15","mpaaRating":"NR","synopsis":"HELD FOR RANSOM tells the true story of Danish photojournalist Daniel Rye who was held hostage for 398 days in Syria by the terror organization ISIS along with several other foreign nationals including the American journalist, James Foley. The film follows Daniel's struggle to survive in captivity, his friendship with James, and the nightmare of the Rye family back home in Denmark as they try to do everything in their power to save their son. At the center of this crisis, we find hostage negotiator, Arthur, who plays a pivotal role in securing Daniel's release.","synopsisType":"synopsis","runtime":"2 hr. 19 min.","posters":{"thumborId":"v1.bTsxMzkyMjgxNjtqOzE5MDE2OzEyMDA7MTk0MjsyODc3","primary":"https://resizing.flixster.com/1_Kjzro1SMbTz79NauPbApQxrqs=/130x0/v1.bTsxMzkyMjgxNjtqOzE5MDE2OzEyMDA7MTk0MjsyODc3"},"actors":["Esben Smed", "Anders W. Berthelsen", "Toby Kebbell"]},{"id":771512626,"title":"Endless Night (Longa noite)","url":"/m/endless_night_2019","tomatoIcon":"fresh","tomatoScore":88,"dvdReleaseDate":"Oct 20","mpaaRating":"NR","runtime":"1 hr. 32 min.","posters":{"thumborId":"v1.bTsxMzk0MTYyNDtqOzE5MDE2OzEyMDA7MjE2MDsyODgw","primary":"https://resizing.flixster.com/iXiD7X83naYEZ2YQB9hWI8SvSXc=/130x0/v1.bTsxMzk0MTYyNDtqOzE5MDE2OzEyMDA7MjE2MDsyODgw"},"actors":["Misha Bies Golas", "Nuria Lestegás", "Manuel 'Pozas' Vázquez"]},{"id":771512694,"title":"The Capote Tapes","url":"/m/the_capote_tapes","tomatoIcon":"fresh","tomatoScore":92,"theaterReleaseDate":"Sep 10","dvdReleaseDate":"Oct 26","mpaaRating":"NR","synopsis":"Although its rather workmanlike approach may be an awkward fit, <em>The Capote Tapes</em> offers a consistently engaging primer for its sublime subject.","synopsisType":"consensus","runtime":"1 hr. 38 min.","posters":{"thumborId":"v1.bTsxMzkwNzAwMDtqOzE5MDE1OzEyMDA7MTk2MjsyODk4","primary":"https://resizing.flixster.com/81fBAGheU_JRfDog7_GAEbv_YCs=/130x0/v1.bTsxMzkwNzAwMDtqOzE5MDE1OzEyMDA7MTk2MjsyODk4"},"actors":["George Plimpton"]},{"id":771515415,"title":"In Full Bloom","url":"/m/in_full_bloom","tomatoIcon":"fresh","tomatoScore":88,"theaterReleaseDate":"Oct 15","dvdReleaseDate":"Oct 15","mpaaRating":"NR","synopsis":"In post-WWII Tokyo, Japan's undefeated boxing champion, Masahiro (Yusuke Ogasawara) trains in the winter wilderness for his upcoming battle against the American challenger, Clint Sullivan (Tyler Wood). Sullivan, who's haunted by memories of the war, must overcome the Yakuza's influence to preserve his honor. Pitted against political tensions, the fighters' parallel journeys will test the very limits of the human spirit.","synopsisType":"synopsis","runtime":"1 hr. 29 min.","posters":{"thumborId":"v1.bTsxMzk0OTMyMTtwOzE5MDE2OzEyMDA7MTIwMDsxNzc2","primary":"https://resizing.flixster.com/CtpqYoJc9ASusf_Aj85uf_ckZ7E=/130x0/v1.bTsxMzk0OTMyMTtwOzE5MDE2OzEyMDA7MTIwMDsxNzc2"},"actors":["Tyler Wood", "Yusuke Ogasawara", "S. Scott McCracken"]},{"id":771520519,"title":"De Gaulle","url":"/m/de_gaulle","tomatoIcon":"rotten","tomatoScore":40,"theaterReleaseDate":"Oct 22","dvdReleaseDate":"Oct 22","mpaaRating":"NR","synopsis":"&#8200;May 1940. France is facing a disastrous military situation against the German army. Charles de Gaulle, newly appointed General, joins the Government in Paris while Yvonne, his wife, and their three children stay in the East. Facing the defeatist attitude of Pétain ready to negotiate with Hitler, de Gaulle has one purpose: continue fighting. And along with thousands of French families, Yvonne and the children are soon forced to flee the advancing German troops. Without contact from one another, the doubt arises: will the de Gaulle family be sacrificed for the sake of France?","synopsisType":"synopsis","runtime":"1 hr. 48 min.","posters":{"thumborId":"v1.bTsxMzgwNTM2NjtqOzE5MDE0OzEyMDA7OTQ0OzEyODA","primary":"https://resizing.flixster.com/JFuvveNP5Z3Poq8kFVU3skJ_pIg=/130x0/v1.bTsxMzgwNTM2NjtqOzE5MDE0OzEyMDA7OTQ0OzEyODA"},"actors":["Lambert Wilson", "Isabelle Carré", "Olivier Gourmet"]},{"id":771535048,"title":"Keyboard Fantasies","url":"/m/keyboard_fantasies","tomatoIcon":"fresh","tomatoScore":100,"theaterReleaseDate":"Oct 29","dvdReleaseDate":"Oct 29","mpaaRating":"NR","synopsis":"Keyboard Fantasies tells the story of Beverly Glenn-Copeland, a black transgender septuagenarian (and musical genius) who finally finds his place in the world. When Glenn receives an unexpected email in 2016 from a record collector in Japan enquiring about copies of his 1986 self-release, Keyboard Fantasies, everything changes. Now signed to a major indie label, and sharing a timely message with the world, Glenn's emergence from obscurity transpires as an intimate coming of age story that spins the pain and suffering of prejudice into rhythm, hope and joy.","synopsisType":"synopsis","runtime":"1 hr. 3 min.","posters":{"thumborId":"v1.bTsxMzk0ODQ2NjtqOzE5MDE2OzEyMDA7ODE3NTsxMjA3NQ","primary":"https://resizing.flixster.com/3pNnqb1p-nrgLkrP7Ezy8cXS5ks=/130x0/v1.bTsxMzk0ODQ2NjtqOzE5MDE2OzEyMDA7ODE3NTsxMjA3NQ"},"actors":["Beverly Glenn-Copeland"]},{"id":771535516,"title":"The Estate","url":"/m/the_estate_2021","tomatoIcon":"rotten","tomatoScore":33,"theaterReleaseDate":"Oct 22","dvdReleaseDate":"Oct 22","mpaaRating":"R","synopsis":"First-time feature director James Kapner deftly balances humor and horror, using his keen eye to develop a perfectly campy, Ryan Murphy-esque world. When a narcissistic son (Chris Baker) yearning for a life of luxury and his father's erratic gold-digging wife (Eliza Coupe) decide to kill their way into their inheritance, they employ the help of an absurdly handsome, mysterious hitman (Greg Finley), initiating a psychosexual love triangle that spirals into more than anyone bargained for.","synopsisType":"synopsis","runtime":"1 hr. 25 min.","posters":{"thumborId":"v1.bTsxMzk0MzI0NztqOzE5MDE2OzEyMDA7NDA1Mjs1OTgw","primary":"https://resizing.flixster.com/wNqlUGJE73YNy0ajpkY08A32XO0=/130x0/v1.bTsxMzk0MzI0NztqOzE5MDE2OzEyMDA7NDA1Mjs1OTgw"},"actors":["Chris Baker", "Eliza Coupe", "Christopher Charles Baker"]},{"id":771555202,"title":"Knocking","url":"/m/knocking_2021","tomatoIcon":"certified_fresh","tomatoScore":80,"theaterReleaseDate":"Oct 8","dvdReleaseDate":"Oct 19","mpaaRating":"NR","synopsis":"A slow-burning thriller that teeters between reality and delusion, <em>Knocking</em> views social issues through a blurred lens streaked with horror.","synopsisType":"consensus","runtime":"1 hr. 18 min.","posters":{"thumborId":"v1.bTsxMzk0MzI0MztqOzE5MDE2OzEyMDA7ODI1MDsxMjE1MA","primary":"https://resizing.flixster.com/nmzrNMb5fPsQvbxbb6FMEXx0SEQ=/130x0/v1.bTsxMzk0MzI0MztqOzE5MDE2OzEyMDA7ODI1MDsxMjE1MA"},"actors":["Cecilia Milocco", "Albin Grenholm", "Alexander Salzberger"]},{"id":771555331,"title":"Bergman Island","url":"/m/bergman_island_2021","tomatoIcon":"certified_fresh","tomatoScore":86,"theaterReleaseDate":"Oct 15","dvdReleaseDate":"Oct 22","mpaaRating":"NR","synopsis":"Minor but charming, the well-acted <em>Bergman Island</em> uses the titular filmmaker's legacy as the launchpad for a dreamlike rumination on romance and creativity.","synopsisType":"consensus","runtime":"1 hr. 52 min.","posters":{"thumborId":"v1.bTsxMzkyNjEyMTtqOzE5MDE2OzEyMDA7ODEwMjsxMTk5OQ","primary":"https://resizing.flixster.com/LyT96ZT3ygt1ndOUXt9TM6_Q55o=/130x0/v1.bTsxMzkyNjEyMTtqOzE5MDE2OzEyMDA7ODEwMjsxMTk5OQ"},"actors":["Mia Wasikowska", "Tim Roth", "Vicky Krieps"]},{"id":771555590,"title":"At the Ready","url":"/m/at_the_ready","tomatoIcon":"fresh","tomatoScore":88,"theaterReleaseDate":"Oct 22","dvdReleaseDate":"Oct 22","mpaaRating":"PG13","synopsis":"<em>At the Ready</em> takes the audience inside a fascinating -- and often unsettling -- collision between politics, education, and law enforcement.","synopsisType":"consensus","runtime":"1 hr. 36 min.","posters":{"thumborId":"v1.bTsxMzk2ODE0MDtqOzE5MDE2OzEyMDA7MjE2MDsyODgw","primary":"https://resizing.flixster.com/eOMtRYbZoIq852uL4ovOpYlYQKU=/130x0/v1.bTsxMzk2ODE0MDtqOzE5MDE2OzEyMDA7MjE2MDsyODgw"}},{"id":771561283,"title":"Broadcast Signal Intrusion","url":"/m/broadcast_signal_intrusion","tomatoIcon":"fresh","tomatoScore":71,"theaterReleaseDate":"Oct 22","dvdReleaseDate":"Oct 22","mpaaRating":"NR","synopsis":"<em>Broadcast Signal Intrusion</em> struggles to satisfactorily resolve its setup, but for much of its runtime, it offers an intriguing, well-acted diversion for horror fans.","synopsisType":"consensus","runtime":"1 hr. 44 min.","posters":{"thumborId":"v1.bTsxMzkxMzAzMjtqOzE5MDE2OzEyMDA7MTk0NDsyODgw","primary":"https://resizing.flixster.com/v91mNncCLbIbvoznabtQvpC2Dwk=/130x0/v1.bTsxMzkxMzAzMjtqOzE5MDE2OzEyMDA7MTk0NDsyODgw"},"actors":["Harry Shum Jr.", "Kelley Mack", "Anthony Cabral"]},{"id":771561296,"title":"The Spine of Night","url":"/m/the_spine_of_night","tomatoIcon":"fresh","tomatoScore":74,"theaterReleaseDate":"Oct 29","dvdReleaseDate":"Oct 29","mpaaRating":"R","synopsis":"With a hard fantasy story that stands in service of its eye-catching animation, <em>The Spine of Night</em> is a distinctive treat for genre enthusiasts.","synopsisType":"consensus","runtime":"1 hr. 33 min.","posters":{"thumborId":"v1.bTsxMzk2MzQyMDtqOzE5MDE2OzEyMDA7Mjc2NDs0MDk2","primary":"https://resizing.flixster.com/eigUOTLv7V2mCNkseE0KhA2CPZw=/130x0/v1.bTsxMzk2MzQyMDtqOzE5MDE2OzEyMDA7Mjc2NDs0MDk2"},"actors":["Richard E. Grant", "Lucy Lawless", "Patton Oswalt"]},{"id":771561309,"title":"Introducing, Selma Blair","url":"/m/introducing_selma_blair","tomatoIcon":"fresh","tomatoScore":100,"theaterReleaseDate":"Oct 15","dvdReleaseDate":"Oct 21","mpaaRating":"NR","synopsis":"<em>Introducing, Selma Blair</em> lives up to its title with a personal look at a celebrity pulling back the curtain with courage, humor, and grace.","synopsisType":"consensus","runtime":"1 hr. 30 min.","posters":{"thumborId":"v1.bTsxMzk3MDgzODtqOzE5MDE2OzEyMDA7MjE2MDsyODgw","primary":"https://resizing.flixster.com/PbxeQt53kOgWUZCjeLAm9-ZOYNY=/130x0/v1.bTsxMzk3MDgzODtqOzE5MDE2OzEyMDA7MjE2MDsyODgw"},"actors":["Selma Blair"]},{"id":771561603,"title":"Women is Losers","url":"/m/women_is_losers","tomatoIcon":"fresh","tomatoScore":70,"dvdReleaseDate":"Oct 25","mpaaRating":"NR","synopsis":"While it might have benefited from a subtler approach to its message, <em>Women Is Losers</em> is elevated by outstanding work from lead Lorenza Izzo.","synopsisType":"consensus","runtime":"1 hr. 24 min.","posters":{"thumborId":"v1.bTsxMzc0MTc2NTtqOzE5MDE0OzEyMDA7NDUwOzY2MA","primary":"https://resizing.flixster.com/JzDRHtQtwHudud3KNNCeYV0zd5Q=/130x0/v1.bTsxMzc0MTc2NTtqOzE5MDE0OzEyMDA7NDUwOzY2MA"},"actors":["Lorenza Izzo"]},{"id":771562024,"title":"Tango Shalom","url":"/m/tango_shalom","tomatoIcon":"fresh","tomatoScore":73,"theaterReleaseDate":"Sep 3","dvdReleaseDate":"Oct 29","mpaaRating":"NR","synopsis":"Moshe Yehuda (Jos Laniado), a Hasidic Rabbi and amateur Hora dancer, enters a big televised Tango competition to save his Hebrew school from bankruptcy. There is only one problem: due to his orthodox religious beliefs, he is not allowed to touch a woman! At odds with his wife and five kids, the Grand Rabbi of his orthodox sect, and Moshe's entire Hasidic community in Crown Heights, Brooklyn, Moshe is forced to ask a Catholic priest, a Muslim imam, and a Sikh holy man for advice. Together, they hash out a plan to help Moshe dance in the Tango contest \"without sacrificing his sacred beliefs, setting in motion a fun, passionate dance movie. Heart-pumping and heartwarming, \"TANGO SHALOM\" tests the bonds of family and community, and the bounds of tolerance and faith.","synopsisType":"synopsis","runtime":"1 hr. 55 min.","posters":{"thumborId":"v1.bTsxMzg5NTU3MztqOzE5MDE1OzEyMDA7MTQwNjsyMDQ4","primary":"https://resizing.flixster.com/z0FBGastcR0POnnsCoY196nvylA=/130x0/v1.bTsxMzg5NTU3MztqOzE5MDE1OzEyMDA7MTQwNjsyMDQ4"},"actors":["Lainie Kazan", "Jos Laniado", "Judi Beecher"]},{"id":771562522,"title":"The Velvet Underground","url":"/m/the_velvet_underground_2021","tomatoIcon":"certified_fresh","tomatoScore":97,"theaterReleaseDate":"Oct 15","dvdReleaseDate":"Oct 15","mpaaRating":"NR","synopsis":"<em>The Velvet Underground</em> takes a fittingly idiosyncratic approach to delivering a rock documentary that captures the band as well as its era.","synopsisType":"consensus","runtime":"1 hr. 50 min.","posters":{"thumborId":"v1.bTsxMzkxODU2NjtwOzE5MDE2OzEyMDA7MjAwMDszMDAw","primary":"https://resizing.flixster.com/tHcwg5npOok8ZBeF_8qxDX80n7E=/130x0/v1.bTsxMzkxODU2NjtwOzE5MDE2OzEyMDA7MjAwMDszMDAw"},"actors":["Mary Woronov"]},{"id":771562773,"title":"After We Fell","url":"/m/after_we_fell","tomatoIcon":"rotten","tomatoScore":11,"theaterReleaseDate":"Sep 30","dvdReleaseDate":"Oct 19","mpaaRating":"R","synopsis":"The third installment of the \"After\" franchise finds Tessa starting an exciting new chapter of her life. But as she prepares to move to Seattle for her dream job, Hardin's jealousy and unpredictable behavior reach a fever pitch and threaten to end their intense relationship. Their situation grows more complicated when Tessa's father returns and shocking revelations about Hardin's family come to light. Ultimately, Tessa and Hardin must decide if their love is worth fighting for or if it's time to go their separate ways.","synopsisType":"synopsis","posters":{"thumborId":"v1.bTsxMzg2OTM2NTtqOzE5MDE1OzEyMDA7NDA1MDs2MDAw","primary":"https://resizing.flixster.com/En9OPPCiMqOw6BPNljWYyfNUzGE=/130x0/v1.bTsxMzg2OTM2NTtqOzE5MDE1OzEyMDA7NDA1MDs2MDAw"},"actors":["Josephine Langford", "Hero Fiennes Tiffin", "Arielle Kebbel"]},{"id":771562973,"title":"Finding Kendrick Johnson","url":"/m/finding_kendrick_johnson","tomatoIcon":"fresh","tomatoScore":100,"theaterReleaseDate":"Oct 29","dvdReleaseDate":"Oct 29","mpaaRating":"NR","synopsis":"On January 11th, 2013, Kendrick Johnson was found dead in his high school gymnasium rolled up in a gym mat. 'FINDING KENDRICK JOHNSON' is the feature documentary product of a 4 year undercover investigation into the facts of this case. From the creator of 'Stranger Fruit', this new documentary hopes to shed light on one of the most important American stories of our time. So what really happened to KJ? Told through the eyes of KJ's family and close friends, Narrated by Hollywood legend, Jenifer Lewis, Directed by 'Stranger Fruit' creator, Jason Pollock, with an amazing team of Producers including Actor Hill Harper, and Space Jam Director, Malcolm D. Lee. 'FINDING KENDRICK JOHNSON' shares this truly historic, heartbreaking, and unbelievable story with the world for the first time.","synopsisType":"synopsis","runtime":"1 hr. 42 min.","posters":{"thumborId":"v1.bTsxMzg4NjM0OTtqOzE5MDE1OzEyMDA7ODEwMDsxMjAwMg","primary":"https://resizing.flixster.com/S_hkJglPW_GkTpytuuL1o2KA-Bc=/130x0/v1.bTsxMzg4NjM0OTtqOzE5MDE1OzEyMDA7ODEwMDsxMjAwMg"},"actors":["Mitch Credle", "Mitch Credle", "Jackie Johnson"]},{"id":771563043,"title":"Snakehead","url":"/m/snakehead","tomatoIcon":"fresh","tomatoScore":70,"theaterReleaseDate":"Oct 29","dvdReleaseDate":"Oct 29","mpaaRating":"NR","synopsis":"Sister Tse comes to New York through a Snakehead, a human smuggler. She gains favor with the matriarch of the family of crime and she rises the ranks quickly. Soon Tse must reconcile her success with her real reason for coming to America.","synopsisType":"synopsis","runtime":"1 hr. 29 min.","posters":{"thumborId":"v1.bTsxMzkxMTAzMDtqOzE5MDE2OzEyMDA7MjAyNTszMDAw","primary":"https://resizing.flixster.com/k4lZkZFrJYPQC_sblLi4Uo7bgrk=/130x0/v1.bTsxMzkxMTAzMDtqOzE5MDE2OzEyMDA7MjAyNTszMDAw"},"actors":["Sung Kang", "Celia Au"]},{"id":771563352,"title":"Mothers of the Revolution","url":"/m/mothers_of_the_revolution","tomatoIcon":"fresh","tomatoScore":100,"dvdReleaseDate":"Oct 19","mpaaRating":"NR","synopsis":"In 1981 a group of 36 women set off on a 120 mile march from Cardiff to Berkshire to protest against the planned arrival of American nuclear missiles on UK soil. In doing so they started something extraordinary, in time galvanizing over 70,000 women into action to protect their children and future generations. This is the untold story of those Greenham Common women. A tale of how a small number of them made connections with their counterparts in the peace movement behind the Iron Curtain, travelling to the Soviet Union to advance peace and, eventually, contributing towards the end of the Cold War. How they amplified and took the achievements of the Greenham Common movement onto the world stage by being daring, fun, inventive and brave. What they started that day in 1981, around a kitchen table while their children played, became a global movement and the start of a revolution that changed the world.","synopsisType":"synopsis","posters":{"thumborId":"v1.bTsxMzk2MzgxMDtqOzE5MDE2OzEyMDA7MTUzNjsyMDQ4","primary":"https://resizing.flixster.com/p8GjXZXM3uwQLfH8zA5A9FJg4ko=/130x0/v1.bTsxMzk2MzgxMDtqOzE5MDE2OzEyMDA7MTUzNjsyMDQ4"},"actors":["Glenda Jackson"]},{"id":771563401,"title":"Joy Ride","url":"/m/joy_ride_2021","tomatoIcon":"fresh","tomatoScore":100,"theaterReleaseDate":"Oct 29","dvdReleaseDate":"Oct 29","mpaaRating":"NR","synopsis":"Frenemies and veteran comedians Dana Gould and Bobcat Goldthwait, having learned very little from their near-fatal car accident, get back on the road and journey throughout the American South. The documentary captures the duo as they carefully navigate highways and their decades-old contentious friendship; reflecting upon their careers and relationship with comedy. Buckle up.","synopsisType":"synopsis","runtime":"1 hr. 10 min.","posters":{"thumborId":"v1.bTsxMzkyNTc5MjtqOzE5MDE2OzEyMDA7Mjk3Mzs0NTAw","primary":"https://resizing.flixster.com/6n_7RAC5BpHUwV1SuxPHLTkZqpk=/130x0/v1.bTsxMzkyNTc5MjtqOzE5MDE2OzEyMDA7Mjk3Mzs0NTAw"},"actors":["Bobcat Goldthwait", "Dana Gould"]},{"id":771563411,"title":"Warning","url":"/m/warning_2021","tomatoIcon":"rotten","tomatoScore":33,"theaterReleaseDate":"Oct 22","dvdReleaseDate":"Oct 22","mpaaRating":"R","synopsis":"Set in the not-too-distant future, this intense sci-fi thriller explores the repercussions that humanity faces when omniscient technology becomes a substitute for human contact. But life begins to unravel when a global storm causes electronics to go haywire, leading to terrifying, deadly consequences.","synopsisType":"synopsis","posters":{"thumborId":"v1.bTsxMzkyNjE4NDtqOzE5MDE2OzEyMDA7NjAwOzg5MA","primary":"https://resizing.flixster.com/Mpo-Y9_H_fFGZT3NDr9jAhXJvNM=/130x0/v1.bTsxMzkyNjE4NDtqOzE5MDE2OzEyMDA7NjAwOzg5MA"},"actors":["Alice Eve", "Annabelle Wallis", "Thomas Jane"]},{"id":771563427,"title":"Runt","url":"/m/runt_2021","tomatoIcon":"fresh","tomatoScore":60,"theaterReleaseDate":"Oct 1","dvdReleaseDate":"Oct 19","mpaaRating":"PG","synopsis":"Cal (Cameron Boyce) and Cecily (Nicole Elizabeth Berger) are bullied high school students who turn to revenge to settle scores with their tormentors. With no one to turn to, they spiral into a downward cycle of misguided violence.","synopsisType":"synopsis","runtime":"1 hr. 35 min.","posters":{"thumborId":"v1.bTsxMzkyNzI5MztqOzE5MDE2OzEyMDA7MjAwMDszMDAw","primary":"https://resizing.flixster.com/8aVHt9_XjzxKV7Z1hiwA9RZJvrg=/130x0/v1.bTsxMzkyNzI5MztqOzE5MDE2OzEyMDA7MjAwMDszMDAw"},"actors":["Cameron Boyce", "Brianna Hildebrand"]},{"id":771563484,"title":"Army of Thieves","url":"/m/army_of_thieves","tomatoIcon":"fresh","tomatoScore":70,"theaterReleaseDate":"Oct 29","dvdReleaseDate":"Oct 29","mpaaRating":"NR","synopsis":"<em>Army of Thieves</em> doesn't reinvent the heist thriller, but director-star Matthias Schweighöfer proves an appealing presence on both sides of the camera.","synopsisType":"consensus","runtime":"2 hr. 7 min.","posters":{"thumborId":"v1.bTsxMzk3MjcwMztqOzE5MDE2OzEyMDA7MTUwMDsyMjIy","primary":"https://resizing.flixster.com/i9u3FT_0k_4CGRdlb03rYFfYePY=/130x0/v1.bTsxMzk3MjcwMztqOzE5MDE2OzEyMDA7MTUwMDsyMjIy"},"actors":["Matthias Schweighöfer"]},{"id":771563576,"title":"The Village Detective: A Song Cycle","url":"/m/the_village_detective_a_song_cycle","tomatoIcon":"fresh","tomatoScore":90,"theaterReleaseDate":"Sep 22","dvdReleaseDate":"Oct 20","mpaaRating":"NR","synopsis":"During the summer of 2016, a fishing boat off the shores of Iceland made a most curious catch: four reels of 35mm film, seemingly of Soviet provenance. Unlike the film find explored in Bill Morrison's Dawson City: Frozen Time, it turned out this discovery wasn't a lost work of major importance, but an incomplete print of a popular comedy starring beloved Russian actor Mihail arov. Does that mean it has no value? Morrison thought not. To him, the heavily water-damaged print, and the way it surfaced, could be seen as a fitting reflection on the life of arov, who loved this role so much that he even co-directed a sequel to it. Morrison uses the story as a jumping off point for his latest meditation on cinema's past, offering a journey into Soviet history and film accompanied by a gorgeous score by Pulitzer and Grammy-winning composer David Lang.","synopsisType":"synopsis","runtime":"1 hr. 21 min.","posters":{"thumborId":"v1.bTsxMzk0MTYxNztqOzE5MDE2OzEyMDA7MTA2MjsxNTAw","primary":"https://resizing.flixster.com/4uzUUZisrqAu2QjzFHsFOpnEmUI=/130x0/v1.bTsxMzk0MTYxNztqOzE5MDE2OzEyMDA7MTA2MjsxNTAw"},"actors":["Mikhail Zharov"]},{"id":771563598,"title":"Night Teeth","url":"/m/night_teeth","tomatoIcon":"rotten","tomatoScore":34,"dvdReleaseDate":"Oct 20","mpaaRating":"NR","synopsis":"<em>Night Teeth</em> has a solid cast and some interesting ideas, but they're all lost in a listlessly told, generally predictable vampire story.","synopsisType":"consensus","posters":{"thumborId":"v1.bTsxMzk0Mjc3NTtqOzE5MDE2OzEyMDA7ODEwNTsxMjAwMQ","primary":"https://resizing.flixster.com/WfXO1HKpmSIXW08apZIE-B7X3Hs=/130x0/v1.bTsxMzk0Mjc3NTtqOzE5MDE2OzEyMDA7ODEwNTsxMjAwMQ"},"actors":["Jorge Lendeborg Jr.", "Debby Ryan", "Lucy Fry"]},{"id":771563678,"title":"Dashcam","url":"/m/dashcam_2021","tomatoIcon":"fresh","tomatoScore":88,"dvdReleaseDate":"Oct 19","mpaaRating":"R","synopsis":"Inspired by Antonioni's Blow-Up, Brian De Palma's Blow Out, and Francis Ford Coppola's The Conversation, Dashcam is a stunning psychological thriller that follows Jake (Eric Tabach, \"Blue Bloods\")--a timid video editor at a local news channel who fantasizes about becoming a reporter. While editing a piece on a routine traffic stop that resulted in the death of a police officer and a major political official (Larry Fessenden, Dementer, Jakob's Wife), Jake is inadvertently sent dashcam video evidence that tells a completely different story. Working alone from his small apartment in NYC, Jake uses his skills as an editor to analyze the footage and piece together the truth behind what actually happened. Has Jake uncovered a conspiracy that he can break on the morning news? Or is he seeing things that aren't really there?","synopsisType":"synopsis","runtime":"1 hr. 22 min.","posters":{"thumborId":"v1.bTsxMzk0ODY2MTtqOzE5MDE2OzEyMDA7MjAwMDszMDAw","primary":"https://resizing.flixster.com/ivyxzXfoZxLnZKqBlCircd-ST6U=/130x0/v1.bTsxMzk0ODY2MTtqOzE5MDE2OzEyMDA7MjAwMDszMDAw"},"actors":["Larry Fessenden", "Zachary Booth", "Eric Tabach"]},{"id":771563742,"title":"Paranormal Activity: Next of Kin","url":"/m/paranormal_activity_next_of_kin","tomatoIcon":"rotten","tomatoScore":22,"theaterReleaseDate":"Oct 29","dvdReleaseDate":"Oct 29","mpaaRating":"R","synopsis":"Sinister supernatural events trouble a family.","synopsisType":"synopsis","runtime":"1 hr. 38 min.","posters":{"thumborId":"v1.bTsxMzk1NzQ0MjtqOzE5MDE2OzEyMDA7MjAyNTszMDAw","primary":"https://resizing.flixster.com/swda5QP8J0lBIdlHp9ry4i4scn8=/130x0/v1.bTsxMzk1NzQ0MjtqOzE5MDE2OzEyMDA7MjAyNTszMDAw"},"actors":["Roland Buck III", "Jill Andre", "Emily Bader"]}]
    #for dic in lst:
    #    movie_url = 'https://www.rottentomatoes.com' + dic.get('url')
    #    read_movie_page(movie_url, reviews)
    #return reviews