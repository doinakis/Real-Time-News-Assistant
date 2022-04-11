'''
@File    :   SPORT24Scrape.py
@Time    :   2022/04/10 19:46:24
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
  :param category: The article category (Currently supported football, basket, tennis)
  :param start_page: From which page to start scrapping
  :param pages: How many pages to scrap
  '''

  if not os.path.exists(f"{directory}/scrapped_links.json"):
    with open(os.path.join(f"{directory}", 'scrapped_links.json'), 'w') as fp:
      fp.write("[]")

  with open(f"{directory}/scrapped_links.json") as fp:
    scrapped_links = json.load(fp)

  sanity_check = directory.split("/")[-1]

  if (sanity_check != category):
    print("folder and category doesn't match")
    return

  if category == "football":
    url = "https://www.sport24.gr/football/"
  elif category == "basket":
    url = "https://www.sport24.gr/basket/"
  elif category == "tennis":
    url = "https://www.sport24.gr/tennis/"
  else:
    print("Wrong category!")
    return
  page = requests.get(url)

  if page.status_code != 200:
    print("Status not 200")
    return

  soup = BeautifulSoup(page.content, "html.parser")

  for i in range(start_page, pages+1):
    page = requests.get(f"{url}/?pages={i}")
    print(f"{url}?p={i}")
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
  cards = soup.find(id="main")
  article_list = cards.find_all("h1", class_="article__title")
  links_list = []
  for article in article_list:
    links = article.find("a")
    links_list.append(links["href"])
  return links_list


def scrape_article(url, category, directory):
  page = requests.get(url)
  json_name = url.split("/")[-1]

  if page.status_code != 200:
    print("Article fetch error.")
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
      "category": category
    }
  }

  with open(f"{directory}/{json_name}.json", 'w') as file:
    json.dump(dict, file)

  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='unboxholics scrapping script.')
  parser.add_argument('--directory', required=True, help='the directory to store the scrapped articles to')
  parser.add_argument('--category', required=True, help='category to scrape')
  parser.add_argument('--start_page', required=True, help='the page to start the scrapping from', type=int)
  parser.add_argument('--stop_page', required=True, help='the last page to scrap', type=int)
  args = parser.parse_args()
  if not os.path.exists(args.directory):
    os.makedirs(args.directory)
  scrape_init(args.directory, args.category, args.start_page, args.stop_page)