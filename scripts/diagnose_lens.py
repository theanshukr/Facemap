import os
import sys
import json
import requests
from dotenv import load_dotenv

sys.path.insert(0, ".")
from src.search_engine.reverse_search import ReverseImageSearchEngine
from src.search_engine.social_extractor import SocialMediaExtractor

load_dotenv()
engine = ReverseImageSearchEngine()
img_path = "sample_images/who.jpg"

print(f"=== Diagnosing Google Lens for {img_path} ===")
target_url = engine._upload_temp_image(img_path)
print("Uploaded image URL:", target_url)

params = {
    "engine": "google_lens",
    "url": target_url,
    "api_key": engine.serpapi_key
}

resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
print("SerpApi HTTP Status:", resp.status_code)

if resp.status_code == 200:
    data = resp.json()
    v_matches = data.get("visual_matches", [])
    e_matches = data.get("exact_matches", [])
    kg = data.get("knowledge_graph", [])
    print(f"Total visual_matches in raw response: {len(v_matches)}")
    print(f"Total exact_matches in raw response: {len(e_matches)}")
    print(f"Knowledge Graph: {len(kg)}")

    reddit_raw = []
    social_raw = []

    for idx, item in enumerate(v_matches):
        link = item.get("link", "")
        title = item.get("title", "")
        source = item.get("source", "")
        thumb = item.get("thumbnail") or item.get("original")
        
        is_reddit = "reddit.com" in link.lower() or "reddit" in source.lower()
        if is_reddit:
            reddit_raw.append({"index": idx, "link": link, "title": title, "source": source, "has_thumb": bool(thumb)})
        
        is_soc = engine.is_genuine_social_media(link)
        if is_soc:
            social_raw.append({"index": idx, "platform": source, "link": link, "title": title, "has_thumb": bool(thumb)})

    print("\n--- REDDIT DISCOVERY IN RAW RESPONSE ---")
    if reddit_raw:
        print(f"Reddit results present in raw response: YES ({len(reddit_raw)} found)")
        for r in reddit_raw:
            print(f"  [#{r['index']}] Link: {r['link']} | Title: {r['title']} | Has Thumb: {r['has_thumb']}")
    else:
        print("Reddit results present in raw response: NO")

    print(f"\n--- ALL SOCIAL MEDIA RESULTS IN RAW RESPONSE ({len(social_raw)}) ---")
    for s in social_raw:
        print(f"  [#{s['index']}] Platform: {s['platform']} | Link: {s['link']}")

    print("\n--- ALL RAW VISUAL MATCHES (First 30) ---")
    for idx, item in enumerate(v_matches[:30]):
        print(f"#{idx+1}: {item.get('source')} | {item.get('link')} | {item.get('title')[:60] if item.get('title') else ''}")
else:
    print("Failed to query SerpApi:", resp.text)
