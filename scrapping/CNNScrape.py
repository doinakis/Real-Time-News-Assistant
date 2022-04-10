'''
@File    :   CNNScrape.py
@Time    :   2022/04/10 19:46:37
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   None
'''
import requests, json, time, argparse, os
from bs4 import BeautifulSoup

def scrape_init(directory, category, start_page=0, pages=10):
  '''
  Scrape sport related articles from www.sport24.gr
  :param directory: The directory to store the scrapped articles
  :param category: The article category (Currently supported politiki)
  :param start_page: From which page to start scrapping
  :param pages: How many pages to scrap
  '''

  with open(f"{directory}/scrapped_links.json") as fp:
    scrapped_links = json.load(fp)

  sanity_check = directory.split("/")[-1]

  if (sanity_check != category):
    print("folder and category doesn't match")
    return

  url = "https://cnn.gr/politiki"

  for i in range(start_page, pages+1):
    page = requests.get(f"{url}?page={i}")
    print(f"{url}?page={i}")
    if page.status_code != 200:
      print("Status not 200")
      return

    soup = BeautifulSoup(page.content, "html.parser")
    links_list = article_list(soup)
    for link in links_list:
      if link in scrapped_links:
        print("Already in")
      else:
        scrapped_links.append(link)
        scrape_article(link, category, directory)
        print("wait")
        with open(f"{directory}/scrapped_links.json", 'w') as file:
          json.dump(scrapped_links, file)
        time.sleep(3)
    time.sleep(2)

  return


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


def scrape_article(url, category, directory):
  page = requests.get(url)
  json_name = url.split("/")[-1]
  if page.status_code != 200:
    print("Article fetch error.")
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

  with open(f"{directory}/{json_name}.json", 'w') as file:
    json.dump(dict, file)

  return


if __name__ == "__name__":
  parser = argparse.ArgumentParser(description='cnn Greece scrapping script.')
  parser.add_argument('--directory', required=True, help='the directory to store the scrapped articles')
  args = parser.parse_args()
  if not os.path.exists(args.directory):
    os.makedirs(args.directory)
  print('hi')