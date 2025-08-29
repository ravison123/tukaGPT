import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_chapter_urls(main_url):
    BASE_URL = main_url.split("/wiki/")[0]

    """Fetch all chapter links from the main गाथा page."""
    headers = {
        "User-Agent": "TukaGPT/0.1 (https://github.com/yourusername; contact@example.com)"
    }
    response = requests.get(main_url, headers=headers)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    # All links in the content area
    content = soup.find("div", {"class": "mw-parser-output"})
    chapter_links = content.find_all("a", href=True)

    urls = []
    for link in chapter_links:
        href = link["href"]
        decoded = urllib.parse.unquote(href)  # convert back to Marathi
        if decoded.startswith("/wiki/तुकाराम_गाथा/गाथा"):
            urls.append(urllib.parse.urljoin(BASE_URL, href))

    return urls


def scrape_page(url):
    headers = {
        "User-Agent": "TukaGPT/0.1 (https://github.com/yourusername; contact@example.com)"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract the main text
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        return ""

    # Remove tables, scripts, references
    for tag in content_div.find_all(["table", "sup", "style", "script"]):
        tag.decompose()

    # Extract text
    text = content_div.get_text(separator="\n", strip=True)
    return text


def extract_poems_from_page(page_text, end_of_poem_token='<|endoftext|>'):
    pattern = re.compile(
        r"^[०-९0-9]+\s*\n(.*?)(?=\n[०-९0-9]+\s*$)",
        re.S | re.M
    )

    matches = pattern.findall(page_text)

    poems_list = []
    for i, poem in enumerate(matches, 1):
        if poem.endswith('<poem>'):
            continue
        poems_list.append(poem.strip())

    poems_string = ('\n' + end_of_poem_token + '\n').join(poems_list)

    return poems_string + '\n' + end_of_poem_token, len(poems_list)


def get_data(main_url, filename):
    chapter_urls = get_chapter_urls(main_url)
    print(f"Found {len(chapter_urls)} chapter URLs.")

    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(root_data_dir, exist_ok=True)
    filepath = os.path.join(root_data_dir, filename)

    total_poems_extracted = 0
    for url in chapter_urls:
        text = scrape_page(url)

        poems_string, num_poems = extract_poems_from_page(text)
        total_poems_extracted = total_poems_extracted + num_poems

        with open(filepath, "a", encoding="utf-8") as f:  # 'a' = append mode
            f.write(poems_string)

    logging.debug('Number of poems scraped: {}'.format(total_poems_extracted))


if __name__ == "__main__":
    get_data(main_url="https://mr.wikisource.org/wiki/तुकाराम_गाथा",
             filename='tukaram_gatha_overall_data.txt')