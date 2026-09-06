#!/usr/bin/env python3
"""
Independent On-Chain Verification CLI Tool.
Verifies whether a given image matches an immutable record stored on the blockchain.
"""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.blockchain.verifier import ProofVerifier
from src.config import DEFAULT_NETWORK, REGISTRY_CONTRACT_ADDRESS

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Verify an image proof against the blockchain registry"
    )
    parser.add_argument(
        "receipt_file",
        nargs="?",
        default="",
        help="Optional positional path to saved JSON proof receipt file",
    )
    parser.add_argument(
        "--receipt",
        "-r",
        type=str,
        default="",
        help="Path to saved JSON proof receipt file to verify match record against blockchain",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default="",
        help="Path to the image to verify (optional if receipt is provided)",
    )
    parser.add_argument(
        "--contract",
        "-c",
        type=str,
        default=REGISTRY_CONTRACT_ADDRESS,
        help="Deployed FaceProofRegistry contract address (optional, defaults to .env)",
    )
    parser.add_argument(
        "--tx",
        type=str,
        default="",
        help="Transaction hash containing proof calldata (optional)",
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        default=DEFAULT_NETWORK,
        choices=["polygon_amoy", "sepolia", "local"],
        help="Blockchain network to query",
    )

    args = parser.parse_args()
    receipt_target = args.receipt or args.receipt_file

    if not args.image and not receipt_target and not args.tx:
        console.print("[bold red]Please provide a receipt file (e.g. python verify_proof.py <receipt.json>) or --image <path>[/bold red]")
        sys.exit(1)

    console.print(
        Panel(
            "[bold cyan]✦ Blockchain Match Proof Verifier ✦[/bold cyan]\n"
            "Validating cryptographic match record fingerprint against immutable on-chain state...",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    verifier = ProofVerifier(network_key=args.network, contract_address=args.contract)

    if receipt_target:
        outcome = verifier.verify_match_record(receipt_target, contract_address=args.contract)
        from src.utils.ui_display import print_step5_match_verification
        print_step5_match_verification(outcome)
        sys.exit(0 if outcome.is_valid else 1)
    elif args.tx:
        outcome = verifier.verify_by_tx_hash(args.tx, args.image)
    else:
        outcome = verifier.verify_image_by_contract(args.image, args.contract)

    table = Table(title="[bold green]Blockchain Verification Outcome[/bold green]", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", f"[bold green]{outcome.status}[/bold green]" if outcome.is_valid else f"[bold red]{outcome.status}[/bold red]")
    table.add_row("Blockchain", outcome.network)
    table.add_row("Local Image SHA-256", outcome.local_image_hash)
    table.add_row("On-Chain Image Hash", outcome.on_chain_image_hash or "N/A")
    table.add_row("Social Post URL", outcome.social_post_url or "N/A")
    table.add_row("Similarity Score", outcome.similarity_score)
    if outcome.contract_address:
        table.add_row("Contract Address", outcome.contract_address)

    console.print(table)

    if outcome.is_valid:
        console.print("\n[bold green][+] Record is AUTHENTIC and matches the tamper-evident on-chain record![/bold green]\n")
        sys.exit(0)
    else:
        console.print("\n[bold red][x] Verification FAILED or record does not exist on-chain.[/bold red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
