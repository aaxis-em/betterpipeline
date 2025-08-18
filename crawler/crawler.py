from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from onnxruntime import InferenceSession

from priorityqueue.queue import Priorityqueue
from scamweb.detectscam import detect_isscam
from weight.weight import findweight


##crawler class
class Crawler:
    def __init__(self, seedurl, depth):
        self.seedurl = seedurl
        self.depth = depth
        self.queue = Priorityqueue()
        self.sess = InferenceSession(
            "util/spamweb.onnx", providers=["CPUExecutionProvider"]
        )
        self.visited = set()

    ##html extraction getting links <a>

    def extracthtml(self, url):
        if url is None or url in self.visited or self.depth == 0:
            return
        try:
            html = requests.get(url)
            html.encoding = html.apparent_encoding
            soup = BeautifulSoup(html.content, "html.parser")
            links = soup.find_all("a")

            self.visited.add(url)

            for link in links:
                href = link.get("href")
                if href:
                    full_url = urljoin(url, href)

                    if full_url in self.visited:
                        continue

                    if full_url.startswith("http"):
                        notvalid = detect_isscam(full_url, self.sess)
                        if not notvalid:
                            weight = findweight(full_url)
                            self.queue.enqueue(full_url, weight)
                            self.visited.add(full_url)
            self.depth -= 1
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")

    ##start crawling
    def start_crawl(self):
        self.queue.enqueue(self.seedurl, findweight(self.seedurl))

        while not self.queue.is_empty():
            next_url = self.queue.dequeue()
            print(next_url)
            self.extracthtml(next_url[0])
        print("Crawling complete")
