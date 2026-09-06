"""
Global Configuration for Face ID + Blockchain Verification Pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Blockchain Configuration
DEFAULT_NETWORK = os.getenv("BLOCKCHAIN_NETWORK", "polygon_amoy").lower()

# Multiple public RPC endpoints for failover reliability
POLYGON_AMOY_RPCS = [
    os.getenv("POLYGON_AMOY_RPC_URL", "https://polygon-amoy.drpc.org"),
    "https://polygon-amoy-bor-rpc.publicnode.com",
    "https://rpc-amoy.polygon.technology",
]

NETWORKS = {
    "polygon_amoy": {
        "name": "Polygon Amoy Testnet",
        "chain_id": 80002,
        "rpc_url": POLYGON_AMOY_RPCS[0],
        "rpc_fallback_urls": POLYGON_AMOY_RPCS,
        "explorer_tx_url": "https://amoy.polygonscan.com/tx/",
        "explorer_address_url": "https://amoy.polygonscan.com/address/",
    },
    "sepolia": {
        "name": "Ethereum Sepolia Testnet",
        "chain_id": 11155111,
        "rpc_url": os.getenv("SEPOLIA_RPC_URL", "https://rpc.sepolia.org"),
        "rpc_fallback_urls": [
            os.getenv("SEPOLIA_RPC_URL", "https://rpc.sepolia.org"),
            "https://ethereum-sepolia-rpc.publicnode.com",
            "https://rpc2.sepolia.org",
        ],
        "explorer_tx_url": "https://sepolia.etherscan.com/tx/",
        "explorer_address_url": "https://sepolia.etherscan.com/address/",
    },
    "local": {
        "name": "Local EVM (Hardhat / Anvil)",
        "chain_id": 31337,
        "rpc_url": os.getenv("LOCAL_RPC_URL", "http://127.0.0.1:8545"),
        "rpc_fallback_urls": [os.getenv("LOCAL_RPC_URL", "http://127.0.0.1:8545")],
        "explorer_tx_url": "http://localhost:8545/tx/",
        "explorer_address_url": "http://localhost:8545/address/",
    }
}

# Wallet / Keys
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
REGISTRY_CONTRACT_ADDRESS = os.getenv("REGISTRY_CONTRACT_ADDRESS", "")

# Reverse Image Search Configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY", "")
SCRAPE_DO_TOKEN = os.getenv("SCRAPE_DO_TOKEN", "")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID", "")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")

# Social Media Platforms to look for in reverse-search results
SUPPORTED_SOCIAL_DOMAINS = [
    "x.com",
    "twitter.com",
    "linkedin.com",
    "instagram.com",
    "facebook.com",
    "reddit.com",
    "github.com",
    "youtube.com",
    "threads.net",
    "bsky.app",
    "mastodon.social",
    "medium.com",
    "tiktok.com"
]

# Face Matching Thresholds
# For OpenCV SFace 128-D neural embeddings, cosine similarity >= 0.40 indicates a verified match
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))
