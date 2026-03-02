"""
Type-Moon Wiki Scraper
Scrapes lore content from https://typemoon.fandom.com via the MediaWiki API
and saves raw text as JSON to /data/raw/.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://typemoon.fandom.com/api.php"
RAW_DATA_DIR = Path("data/raw")
REQUEST_DELAY = 1.5  # seconds between requests (be polite)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff base

# Target pages grouped by category
TARGET_PAGES: dict[str, list[str]] = {
    "servant": [
        # Fate/stay night
        "Saber_(Fate/stay_night)",       # Artoria Pendragon
        "Archer_(Fate/stay_night)",       # EMIYA
        "Lancer_(Fate/stay_night)",       # Cu Chulainn
        "Heracles",                       # Berserker
        "Rider_(Fate/stay_night)",        # Medusa
        "Medea",                          # Caster
        "Sasaki_Kojiro",                  # Assassin
        "Hassan-i-Sabbah_(Fate/stay_night)",  # True Assassin (Heaven's Feel)
        "Gilgamesh",
        # Fate/Zero
        "Lancer_(Fate/Zero)",             # Diarmuid
        "Rider_(Fate/Zero)",              # Iskandar
        "Berserker_(Fate/Zero)",          # Lancelot
        "Caster_(Fate/Zero)",             # Gilles de Rais
        "Hassan-i-Sabbah_(Fate/Zero)",    # Hundred-Faced Hassan
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
        "Heaven's_Feel",
        "Unlimited_Blade_Works_(route)",
    ],
}


def _clean_html(html: str) -> str:
    """Extract plain text from rendered wiki HTML, stripping boilerplate."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.select(
        "nav, footer, .navbox, .toc, .mw-editsection, "
        ".infobox, #toc, .noprint, .mw-jump-link, "
        "script, style, [role='navigation'], .portable-infobox, "
        ".references, .reflist, sup.reference"
    ):
        tag.decompose()

    paragraphs = []
    for element in soup.find_all(["p", "h2", "h3", "h4", "li"], recursive=True):
        text = element.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def _fetch_page_text(title: str, session: requests.Session) -> Optional[tuple[str, str]]:
    """
    Fetch a wiki page via the MediaWiki parse API (returns rendered HTML).
    Returns (resolved_title, plain_text) or None on failure.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "text|displaytitle",
        "redirects": "1",
        "format": "json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                code = data["error"].get("code", "")
                if code == "missingtitle":
                    logger.warning(f"Page not found: {title}")
                else:
                    logger.warning(f"API error for {title}: {data['error']}")
                return None

            parse = data.get("parse", {})
            resolved_title = BeautifulSoup(
                parse.get("displaytitle", title), "html.parser"
            ).get_text(strip=True)
            html = parse.get("text", {}).get("*", "")

            if not html:
                logger.warning(f"Empty HTML for: {title}")
                return None

            content = _clean_html(html)
            if not content.strip():
                logger.warning(f"Empty content after cleaning: {title}")
                return None

            return resolved_title, content

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(
                f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    logger.error(f"All retries exhausted for: {title}")
    return None


def scrape_page(slug: str, category: str, session: requests.Session) -> Optional[dict]:
    """Fetch a single wiki page via the API and return structured data."""
    logger.info(f"Fetching [{category}] {slug}")

    result = _fetch_page_text(slug, session)
    if result is None:
        return None

    resolved_title, content = result

    return {
        "title": resolved_title,
        "url": f"https://typemoon.fandom.com/wiki/{slug}",
        "slug": slug,
        "category": category,
        "content": content,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def run_scraper(output_dir: Path = RAW_DATA_DIR) -> list[dict]:
    """
    Fetch all target pages and save results to JSON files.
    Returns list of all scraped documents.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "FateRAG/1.0 (educational project; MediaWiki API client)",
            "Accept": "application/json",
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

            time.sleep(REQUEST_DELAY)

        category_file = output_dir / f"{category}.json"
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(category_docs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(category_docs)} documents to {category_file}")

    combined_file = output_dir / "all_documents.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    logger.info(f"Total: {len(all_docs)} documents saved to {combined_file}")

    return all_docs


if __name__ == "__main__":
    docs = run_scraper()
    print(f"\nScraping complete. {len(docs)} pages collected.")
