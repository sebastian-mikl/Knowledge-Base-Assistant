from playwright.sync_api import sync_playwright
import time
import os
import csv

# Create output directory
SAVE_DIR = "scraped_articles"
os.makedirs(SAVE_DIR, exist_ok=True)

# Starting URL - modify this for different knowledge bases
URL = "https://uni.foodhub.com/knowledge/manual/product/article/my-business-hub-login"

with sync_playwright() as p:
    user_data_dir = "auth_storage"

    print("Launching browser...")
    context = p.chromium.launch_persistent_context(
        user_data_dir,
        headless=False,
    )
    page = context.pages[0] if context.pages else context.new_page()

    print(f"Navigating to: {URL}")
    page.goto(URL)

    print("Scroll manually to load all articles...")
    input("Press ENTER when finished scrolling.\n")

    print("Collecting article links...")
    links = page.eval_on_selector_all(
        "a",
        "elements => elements.map(el => el.href).filter(href => href.includes('/article/'))"
    )

    # Save to CSV
    csv_file = os.path.join(SAVE_DIR, "article_links.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["link"])
        for link in links:
            writer.writerow([link])

    print(f"Saved {len(links)} links to {csv_file}")
    context.close()