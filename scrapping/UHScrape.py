'''
@File    :   UHScrape.py
@Time    :   2022/04/10 19:46:08
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   None
'''
import requests, json, time, argparse, os
from bs4 import BeautifulSoup

def scrape_init(directory, category, start_page, pages=10):
  '''
  Scrape sport related articles from www.unboxholics.com
  :param directory: The directory to store the scrapped articles
  :param category: The article category (Currently supported tech, movies, gamings)
  :param start_page: From which page to start scrapping
  :param pages: How many pages to scrap. If the pages variable is set to -1  it scrapes all the articles
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

  if category == "tech":
    url = "https://unboxholics.com/news/tech"
  elif category == "movies":
    url = "https://unboxholics.com/news/movies"
  elif category == "gaming":
    url = "https://unboxholics.com/news/gaming"
  else:
    print("Wrong category!")
    return

  page = requests.get(url)

  if page.status_code != 200:
    print("Status not 200")
    return

  soup = BeautifulSoup(page.content, "html.parser")
  pages = number_of_pages(soup, pages)

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


def number_of_pages(soup, pages):
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

  if pages == -1 or pages > actual_pages:
    pages = actual_pages

  return pages


def article_list(soup):
  cards = soup.find(id="main-container")
  article_list = cards.find_all("article", class_="entry card post-list")
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