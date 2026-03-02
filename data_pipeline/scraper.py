"""
Type-Moon Wiki Scraper
Scrapes lore content from https://typemoon.fandom.com/wiki/
and saves raw text as JSON to /data/raw/.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://typemoon.fandom.com/wiki/"
RAW_DATA_DIR = Path("data/raw")
REQUEST_DELAY = 1.5  # seconds between requests (be polite)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff base

# Target pages grouped by category
TARGET_PAGES: dict[str, list[str]] = {
    "servant": [
        "Saber_(Fate/stay_night)",
        "Archer_(Fate/stay_night)",
        "Lancer_(Fate/stay_night)",
        "Berserker_(Fate/stay_night)",
        "Rider_(Fate/stay_night)",
        "Caster_(Fate/stay_night)",
        "Assassin_(Fate/stay_night)",
        "Gilgamesh",
        "Cu_Chulainn",
        "Medusa_(Fate/stay_night)",
        "Medea_(Fate/stay_night)",
        "Hassan-i-Sabbah_(Fate/stay_night)",
        "Saber_(Fate/Zero)",
        "Lancer_(Fate/Zero)",
        "Rider_(Fate/Zero)",
        "Berserker_(Fate/Zero)",
        "Caster_(Fate/Zero)",
        "Archer_(Fate/Zero)",
        "Assassin_(Fate/Zero)",
    ],
    "noble_phantasm": [
        "Excalibur",
        "Unlimited_Blade_Works",
        "Gáe_Bolg",
        "Gate_of_Babylon",
        "Caladbolg_II",
        "Rho_Aias",
        "Rule_Breaker",
        "Knight_of_Owner",
        "Ionioi_Hetairoi",
    ],
    "master": [
        "Shirou_Emiya",
        "Rin_Tohsaka",
        "Sakura_Matou",
        "Illyasviel_von_Einzbern",
        "Shinji_Matou",
        "Kirei_Kotomine",
        "Kiritsugu_Emiya",
        "Waver_Velvet",
        "Kariya_Matou",
        "Ryuunosuke_Uryuu",
    ],
    "lore": [
        "Holy_Grail_War",
        "Fuyuki_City",
        "Magic_Circuit",
        "Thaumaturgy",
        "Servant",
        "Command_Spell",
        "Prana",
        "Origin",
    ],
    "route": [
        "Fate/stay_night",
        "Fate/Zero",
        "Fate/Grand_Order",
        "Heaven%27s_Feel",
        "Unlimited_Blade_Works_(route)",
    ],
}


def _make_request(url: str, session: requests.Session) -> Optional[requests.Response]:
    """Make an HTTP GET request with retry logic and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait:.1f}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    logger.error(f"All retries exhausted for URL: {url}")
    return None


def _clean_html(soup: BeautifulSoup) -> str:
    """
    Extract main content from the wiki page, stripping navigation,
    infoboxes, footers, and other boilerplate.
    """
    # Remove unwanted elements
    for tag in soup.select(
        "nav, footer, .navbox, .toc, .mw-editsection, "
        ".infobox, #toc, .noprint, .mw-jump-link, "
        ".page-header, .global-navigation, .fandom-sticky-header, "
        ".WikiaArticleFooter, .page-footer, script, style, "
        "[role='navigation'], .portable-infobox"
    ):
        tag.decompose()

    # Target the main article content
    content_div = (
        soup.select_one(".mw-parser-output")
        or soup.select_one("#mw-content-text")
        or soup.select_one(".page-content")
    )

    if not content_div:
        logger.warning("Could not find main content div, using body text.")
        return soup.get_text(separator="\n", strip=True)

    # Extract text with sensible whitespace
    paragraphs = []
    for element in content_div.find_all(["p", "h2", "h3", "h4", "li"], recursive=True):
        text = element.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def scrape_page(slug: str, category: str, session: requests.Session) -> Optional[dict]:
    """Scrape a single wiki page and return structured data."""
    url = urljoin(BASE_URL, slug)
    logger.info(f"Scraping [{category}] {url}")

    response = _make_request(url, session)
    if response is None:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract page title
    title_tag = soup.select_one("h1.page-header__title") or soup.select_one("#firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else slug.replace("_", " ")

    content = _clean_html(soup)
    if not content.strip():
        logger.warning(f"Empty content for {url}")
        return None

    return {
        "title": title,
        "url": url,
        "slug": slug,
        "category": category,
        "content": content,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def run_scraper(output_dir: Path = RAW_DATA_DIR) -> list[dict]:
    """
    Scrape all target pages and save results to JSON files.
    Returns list of all scraped documents.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "FateRAG-DataPipeline/1.0 "
                "(Educational RAG project; contact: your-email@example.com)"
            )
        }
    )

    all_docs: list[dict] = []

    for category, slugs in TARGET_PAGES.items():
        category_docs: list[dict] = []

        for slug in slugs:
            doc = scrape_page(slug, category, session)
            if doc:
                category_docs.append(doc)
                all_docs.append(doc)

            # Rate limiting
            time.sleep(REQUEST_DELAY)

        # Save per-category file
        category_file = output_dir / f"{category}.json"
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(category_docs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(category_docs)} documents to {category_file}")

    # Save combined file
    combined_file = output_dir / "all_documents.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    logger.info(f"Total: {len(all_docs)} documents saved to {combined_file}")

    return all_docs


if __name__ == "__main__":
    docs = run_scraper()
    print(f"\nScraping complete. {len(docs)} pages collected.")
