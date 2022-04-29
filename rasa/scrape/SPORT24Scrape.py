'''
@File    :   SPORT24Scrape.py
@Time    :   2022/04/10 19:46:24
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Scrape API for Rasa Bot
'''
import requests, json, time, argparse, os, logging
from bs4 import BeautifulSoup

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)

def scrape_sport24(directory, num_of_articles=10, pages=100, articles=[]):
  '''
  Scrape sport related articles from www.unboxholics.com
  :param directory: The directory to store the scrapped articles
  :param num_of_articles: How many articles to scrape
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)
  if not os.path.exists(f"{directory}/scrapped_sport24.json"):
    with open(os.path.join(f"{directory}", 'scrapped_sport24.json'), 'w') as fp:
      fp.write("[]")

  articles += scrape_category(directory=directory, category='football', num_of_articles=num_of_articles, pages=pages)
  articles += scrape_category(directory=directory, category='tennis', num_of_articles=num_of_articles, pages=pages)
  articles += scrape_category(directory=directory, category='basket', num_of_articles=num_of_articles, pages=pages)

  logger.info('Sport24 scrape complete.')

  return articles

def scrape_category(directory, category, num_of_articles, pages):
  with open(f"{directory}/scrapped_sport24.json") as fp:
    scrapped_links = json.load(fp)
  articles = []
  url = f"https://www.sport24.gr/{category}/"
  counter = 0
  page = requests.get(url)
  if page.status_code != 200:
    logger.debug(f"Request status not 200.")
    return
  soup = BeautifulSoup(page.content, "html.parser")

  for i in range(0, pages+1):
    page = requests.get(f"{url}/?pages={i}")
    if page.status_code != 200:
      logger.debug(f"Request status not 200.")
      return
    soup = BeautifulSoup(page.content, "html.parser")
    links_list = article_list(soup)
    for link in links_list:
      if counter == num_of_articles:
        return articles
      if link in scrapped_links:
        pass
      else:
        logger.debug(f"Scrap.")
        counter += 1
        scrapped_links.append(link)
        articles.append(scrape_article(url=link, category=category))
        with open(f"{directory}/scrapped_sport24.json", 'w') as file:
          json.dump(scrapped_links, file)
        time.sleep(3)
    time.sleep(2)

  return


def article_list(soup):
  cards = soup.find(id="main")
  article_list = cards.find_all("h1", class_="article__title")
  links_list = []
  for article in article_list:
    links = article.find("a")
    links_list.append(links["href"])
  return links_list


def scrape_article(url, category):
  page = requests.get(url)
  json_name = url.split("/")[-1]

  if page.status_code != 200:
    logger.debug(f"Request status not 200.")
    return
  soup = BeautifulSoup(page.content, "html.parser")
  cards = soup.find(id="article-container")

  article_content = ""
  paragraphs = cards.find_all("p", class_="article-single__lead")
  for paragraph in paragraphs:
    article_content += paragraph.text.strip()

  paragraphs = cards.find_all("p", class_="")

  for paragraph in paragraphs:
    article_content += paragraph.text.strip()

  title = cards.find_all("h1", class_="article-single__title")
  dict = {
    "content": article_content,
    "meta": {
      "url": url,
      "name": f"{json_name}.json",
      "title": title[0].text.strip(),
      "category": "sport"
    }
  }

  return dict
