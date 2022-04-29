'''
@File    :   CNNScrape.py
@Time    :   2022/04/10 19:46:37
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

def scrape_cnn(directory, num_of_articles=10, pages=100, articles=[]):
  '''
  Scrape sport related articles from www.unboxholics.com
  :param directory: The directory to store the scrapped articles
  :param num_of_articles: How many articles to scrape
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)

  if not os.path.exists(f"{directory}/scrapped_cnn.json"):
    with open(os.path.join(f"{directory}", 'scrapped_cnn.json'), 'w') as fp:
      fp.write("[]")

  with open(f"{directory}/scrapped_cnn.json") as fp:
    scrapped_links = json.load(fp)

  articles += scrape_category(directory=directory, category='politiki', num_of_articles=num_of_articles, pages=pages)

  logger.info('CNN scrape complete.')

  return articles

def scrape_category(directory, category, num_of_articles, pages):
  with open(f"{directory}/scrapped_cnn.json") as fp:
    scrapped_links = json.load(fp)

  articles = []
  url = f'https://cnn.gr/{category}'
  counter = 0

  for i in range(1, pages):
    page = requests.get(f"{url}?page={i}")
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
        counter +=1
        scrapped_links.append(link)
        articles.append(scrape_article(url=link, category=category))
        with open(f"{directory}/scrapped_cnn.json", 'w') as file:
          json.dump(scrapped_links, file)
        time.sleep(3)
    time.sleep(2)
  return articles

def article_list(soup):
  cards = soup.find("div", class_ = "flex-main")
  article_list = cards.find_all("a", class_ = "item-link")
  links_list = []
  for article in article_list:
    if article["href"].find("https://www.cnn.gr/") != -1:
      links = article["href"]
    else:
      links = "https://cnn.gr" + article["href"]
    links_list.append(links)
  return links_list


def scrape_article(url, category):
  page = requests.get(url)
  json_name = url.split("/")[-1]
  if page.status_code != 200:
    logger.debug(f"Request status not 200.")
    return
  soup = BeautifulSoup(page.content, "html.parser")
  cards = soup.find("div", class_ = "main-content story-content")
  title = soup.find("h1", class_ = "main-title").text.strip()
  paragraphs = cards.find_all(["p","h2"])
  article_content = ""
  for paragraph in paragraphs:
    article_content += paragraph.text.strip()

  article_content = title + ". " + article_content

  dict = {
    "content": article_content,
    "meta": {
      "url": url,
      "name": f"{json_name}.json",
      "title": title,
      "category": category
    }
  }

  return dict
