import os
import asyncio
import re
import pandas as pd
from playwright.async_api import async_playwright

# --- CONFIGURATION ---
BASE_DOMAIN = "http://dialects.albanianlanguage.net/"
REGIONS = ["AL", "KS", "MK", "CG", "GR", "HR-IT"]
DOWNLOAD_FOLDER = "albanian_recordings"
USER_DATA_DIR = "./playwright_session"  # Stores cookie 'Accept' choice

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

def clean_name(text):
    """Cleans strings for safe filenames."""
    return "".join([c for c in text if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')

async def get_all_speaker_links(page):
    """Crawls regional index pages to find all valid speaker .htm links."""
    all_tasks = []
    for region in REGIONS:
        region_url = f"{BASE_DOMAIN}{region}/"
        print(f"-> Scanning index: {region_url}")
        try:
            await page.goto(region_url, wait_until="networkidle")
            links = await page.query_selector_all("a")
            for link in links:
                href = await link.get_attribute("href")
                # Filter for speaker pages (e.g., 157.htm, 023ks.htm) 
                # and avoid index/navigation links
                if href and ".htm" in href and "index" not in href.lower():
                    full_url = region_url + href if not href.startswith("http") else href
                    if (region, full_url) not in all_tasks:
                        all_tasks.append((region, full_url))
        except Exception as e:
            print(f"   [!] Error scanning {region}: {e}")
    return all_tasks

async def scrape_speaker(context, region_code, url):
    """Visits a speaker page, extracts metadata, and downloads the audio."""
    page = await context.new_page()
    audio_url = None
    
    # Listener to capture the MP3 URL when the 'Play' button is clicked
    async def intercept_response(response):
        nonlocal audio_url
        if ".mp3" in response.url:
            audio_url = response.url

    page.on("response", intercept_response)

    try:
        await page.goto(url, wait_until="networkidle", timeout=60000)
        
        # 1. Extract Metadata from Xara's coordinate-based spans
        spans = await page.query_selector_all(".xr_tl")
        text_list = []
        for s in spans:
            t = await s.inner_text()
            if t.strip(): text_list.append(t.strip())

        # Logic to pair 'Label' with the 'Value' that follows it
        entry = {
            "region_code": region_code,
            "url": url,
            "dialect": "Unknown",
            "location": "Unknown",
            "speaker_info": "Unknown"
        }
        
        for idx, val in enumerate(text_list):
            if "Dialect" in val and idx + 1 < len(text_list):
                entry["dialect"] = text_list[idx+1]
            if "Location" in val and idx + 1 < len(text_list):
                entry["location"] = text_list[idx+1]
            if "Speaker" in val and idx + 1 < len(text_list):
                entry["speaker_info"] = text_list[idx+1]

        # 2. Trigger the Audio Player
        # We try several possible 'Play' button selectors used by the site
        play_btn = await page.query_selector("//img[contains(@src, 'play')] | //*[contains(@class, 'xr_ap')] | //div[contains(@id, 'play')]")
        if play_btn:
            await play_btn.click()
            await asyncio.sleep(4) # Wait for network handshake
        
        # 3. Download the MP3
        if audio_url:
            # Create a descriptive filename for ML organization
            page_id = url.split('/')[-1].replace('.htm', '')
            safe_dialect = clean_name(entry['dialect'])
            filename = f"{region_code}_{page_id}_{safe_dialect}.mp3"
            filepath = os.path.join(DOWNLOAD_FOLDER, filename)
            
            response_data = await page.request.get(audio_url)
            with open(filepath, "wb") as f:
                f.write(await response_data.body())
            
            print(f"   [+] Saved: {filename}")
            entry["file_name"] = filename
            entry["status"] = "Success"
        else:
            print(f"   [!] Audio link not found for: {url}")
            entry["status"] = "No Audio Found"

        return entry

    except Exception as e:
        print(f"   [!] Error on {url}: {e}")
        return None
    finally:
        await page.close()

async def main():
    async with async_playwright() as p:
        # Launch with a persistent context to save 'Accept Cookies' clicks
        context = await p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=True, # Set to True after you've manually clicked 'Accept' on first run
            slow_mo=500
        )
        
        # Create a temp page to crawl the indexes
        explorer_page = await context.new_page()
        speaker_tasks = await get_all_speaker_links(explorer_page)
        await explorer_page.close()
        
        print(f"\nTotal speakers found to process: {len(speaker_tasks)}")
        
        results = []
        for region, url in speaker_tasks:
            data = await scrape_speaker(context, region, url)
            if data:
                results.append(data)
                # Save progress incrementally in case of crash
                pd.DataFrame(results).to_csv("albanian_ml_dataset.csv", index=False)
        
        await context.close()
        print(f"\n--- PROCESS COMPLETE ---")
        print(f"Data saved to albanian_ml_dataset.csv")
        print(f"Audio files saved to /{DOWNLOAD_FOLDER}")

if __name__ == "__main__":
    asyncio.run(main())