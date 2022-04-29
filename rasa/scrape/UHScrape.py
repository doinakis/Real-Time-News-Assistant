'''
@File    :   UHScrape.py
@Time    :   2022/04/10 19:46:08
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Scrape API for Rasa Bot
'''
import requests, json, time, os, logging
from bs4 import BeautifulSoup

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)

def scrape_uh(directory, num_of_articles=10, articles=[]):
  '''
  Scrape sport related articles from www.unboxholics.com
  :param directory: The directory to store the scrapped articles
  :param num_of_articles: How many articles to scrape
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)

  if not os.path.exists(f"{directory}/scrapped_uh.json"):
    with open(os.path.join(f"{directory}", 'scrapped_uh.json'), 'w') as fp:
      fp.write("[]")

  articles += scrape_category(directory=directory, category='tech', num_of_articles=num_of_articles)
  articles += scrape_category(directory=directory, category='movies', num_of_articles=num_of_articles)
  articles += scrape_category(directory=directory, category='gaming', num_of_articles=num_of_articles)

  logger.info('UH scrape complete.')

  return articles

def scrape_category(directory, category, num_of_articles):
  with open(f"{directory}/scrapped_uh.json") as fp:
    scrapped_links = json.load(fp)

  articles = []

  url = f'https://unboxholics.com/news/{category}'
  page = requests.get(url)

  if page.status_code != 200:
    logger.debug(f"Request status not 200.")
    return

  soup = BeautifulSoup(page.content, "html.parser")
  pages = number_of_pages(soup=soup)

  counter = 0
  for i in range(0, pages+1):
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
        counter += 1
        scrapped_links.append(link)
        articles.append(scrape_article(url=link, category=category))
        with open(f"{directory}/scrapped_uh.json", 'w') as file:
          json.dump(scrapped_links, file)
        time.sleep(3)
    time.sleep(2)

  return articles

def number_of_pages(soup):
  cards = soup.find(id="main-container")
  actual_pages = 0
  pagination_links = cards.find_all("div", class_="pagination-links")
  for pagination_link in pagination_links:
    links = pagination_link.find_all("a")
    for link in links:
      link_url = link["href"]
      needle = link_url.find("page=") + 5
      p = int(link_url[needle:])
      if actual_pages < p:
        actual_pages = p

  return actual_pages


def article_list(soup):
  cards = soup.find(id="main-container")
  article_list = cards.find_all("article", class_="entry card post-list")
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
  cards = soup.find(id="main-container")
  title = cards.find_all("h1", class_="single-post__entry-title")[0].text.strip()
  paragraphs = cards.find_all("p", class_="")

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
