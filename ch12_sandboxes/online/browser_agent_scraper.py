"""
browser_agent_scraper.py — Strip a live page down to what an LLM actually needs.

The problem with feeding raw HTML to an LLM: a typical news article is 500
words of content wrapped in 8,000 tokens of scripts, tracking pixels, nav
menus, and ad containers. You're paying for noise.

This scraper runs Playwright headlessly, lets the page JS execute (so
server-rendered AND client-rendered content is visible), then strips
everything that isn't text: scripts, styles, iframes, SVGs, hidden elements.
What's left is the readable content — the part the model needs.

Run it:
    python browser_agent_scraper.py [url]

Defaults to https://example.com if no URL is provided.

Requires: pip install playwright
          playwright install chromium
"""

import sys
from playwright.sync_api import sync_playwright


def scrape(url: str, wait_ms: int = 1000) -> dict:
    """
    Return the page's visible text and title.

    wait_ms: how long to wait after load for JS to finish rendering.
    Increase for heavily JS-driven pages (SPAs, lazy loaders).
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(wait_ms)

        title = page.title()

        # Remove everything that isn't readable content
        page.evaluate("""() => {
            const noise = ['script', 'style', 'iframe', 'svg', 'noscript',
                           'header', 'footer', 'nav', '[aria-hidden="true"]'];
            noise.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });
        }""")

        # innerText respects CSS visibility — hidden elements are excluded
        text = page.locator("body").inner_text()

        browser.close()

    # Collapse blank lines down to a single blank line
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    return {
        "url": url,
        "title": title,
        "text": cleaned,
        "char_count": len(cleaned),
    }


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(f"Scraping: {url}\n")

    result = scrape(url)

    print(f"Title:      {result['title']}")
    print(f"Characters: {result['char_count']}")
    print(f"\n--- Content preview (first 800 chars) ---\n")
    print(result["text"][:800])

    if result["char_count"] > 800:
        print(f"\n... ({result['char_count'] - 800} more characters)")

    print("\n--- End of preview ---")
    print("\nThis is what you pass to the LLM, not the raw HTML.")


if __name__ == "__main__":
    main()
