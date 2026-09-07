#!/usr/bin/env python3
"""
Inspect all data and proof records stored on the blockchain smart contract.
"""

import os
import sys
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from web3 import Web3

from src.config import NETWORKS, DEFAULT_NETWORK, REGISTRY_CONTRACT_ADDRESS
from src.blockchain.contract_abi import FACE_PROOF_REGISTRY_ABI
from src.blockchain.verifier import ProofVerifier

console = Console()


def inspect_blockchain(network_key: str = DEFAULT_NETWORK, contract_address: str = ""):
    target_contract = contract_address or os.getenv("REGISTRY_CONTRACT_ADDRESS", "") or REGISTRY_CONTRACT_ADDRESS
    net_config = NETWORKS.get(network_key, NETWORKS["polygon_amoy"])
    
    console.print(
        Panel(
            f"[bold cyan]✦ Blockchain Data Inspector (FaceProofRegistry) ✦[/bold cyan]\n"
            f"[white]Network:[/white] [bold yellow]{net_config['name']}[/bold yellow] (Chain ID: {net_config['chain_id']})\n"
            f"[white]Contract Address:[/white] [bold green]{target_contract}[/bold green]\n"
            f"[white]Explorer URL:[/white] [blue underline]{net_config['explorer_address_url']}{target_contract}#readContract[/blue underline]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    if not target_contract or not Web3.is_address(target_contract):
        console.print("[bold red]❌ Invalid or missing REGISTRY_CONTRACT_ADDRESS.[/bold red]")
        return

    verifier = ProofVerifier(network_key=network_key, contract_address=target_contract)
    w3 = verifier.w3

    if not w3.is_connected():
        console.print("[bold red]❌ Failed to connect to blockchain RPC endpoint.[/bold red]")
        return

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(target_contract),
        abi=FACE_PROOF_REGISTRY_ABI,
    )

    try:
        total_proofs = contract.functions.getTotalProofs().call()
    except Exception as e:
        console.print(f"[bold red]❌ Failed to read contract: {e}[/bold red]")
        return

    console.print(f"\n[bold white]Total Registered Proofs On-Chain:[/bold white] [bold green]{total_proofs}[/bold green]\n")

    if total_proofs == 0:
        console.print("[yellow]No proof records currently stored in this smart contract.[/yellow]")
        return

    table = Table(
        title="[bold cyan]Immutable On-Chain Records[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Match Record Hash (Key)", style="green", overflow="fold")
    table.add_column("Social Post URL", style="blue", overflow="fold")
    table.add_column("Similarity", style="yellow", justify="center")
    table.add_column("Input Image Hash", style="dim white", overflow="fold")
    table.add_column("Candidate Hash", style="dim white", overflow="fold")
    table.add_column("Timestamp", style="white", max_width=20)
    table.add_column("Verifier Address", style="magenta", overflow="fold")

    records_detail = []

    for i in range(total_proofs):
        try:
            record_hash_bytes = contract.functions.getProofHashByIndex(i).call()
            res = contract.functions.getProof(record_hash_bytes).call()
            exists, on_input_hash, on_cand_hash, on_url, sim_score, on_meta, ts, verifier_addr = res

            rec_hash_hex = "0x" + record_hash_bytes.hex()
            input_hash_hex = "0x" + on_input_hash.hex()
            cand_hash_hex = "0x" + on_cand_hash.hex()
            sim_str = f"{sim_score / 100.0:.2f}%"
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"

            table.add_row(
                str(i + 1),
                rec_hash_hex,
                on_url,
                sim_str,
                input_hash_hex,
                cand_hash_hex,
                date_str,
                verifier_addr,
            )

            records_detail.append({
                "index": i + 1,
                "match_record_hash": rec_hash_hex,
                "input_image_hash": input_hash_hex,
                "candidate_content_hash": cand_hash_hex,
                "social_post_url": on_url,
                "similarity_score": sim_str,
                "timestamp": date_str,
                "verifier": verifier_addr,
                "metadata_raw": on_meta,
            })
        except Exception as err:
            console.print(f"[red]Error fetching record #{i+1}: {err}[/red]")

    console.print(table)

    # Print detailed JSON breakdown for each
    console.print("\n[bold cyan]Detailed Record Payloads (On-Chain Metadata):[/bold cyan]")
    for rec in records_detail:
        meta_formatted = rec["metadata_raw"]
        try:
            parsed = json.loads(rec["metadata_raw"])
            meta_formatted = json.dumps(parsed, indent=2)
        except Exception:
            pass

        console.print(
            Panel(
                f"[bold green]Record #{rec['index']}[/bold green]\n"
                f"[bold]Match Record Hash:[/bold] {rec['match_record_hash']}\n"
                f"[bold]Input Image SHA-256:[/bold] {rec['input_image_hash']}\n"
                f"[bold]Candidate Image SHA-256:[/bold] {rec['candidate_content_hash']}\n"
                f"[bold]Social Post URL:[/bold] {rec['social_post_url']}\n"
                f"[bold]Similarity Score:[/bold] {rec['similarity_score']}\n"
                f"[bold]Timestamp:[/bold] {rec['timestamp']}\n"
                f"[bold]Verifier Wallet:[/bold] {rec['verifier']}\n"
                f"[bold]Metadata JSON:[/bold]\n{meta_formatted}",
                box=box.SIMPLE,
                border_style="dim cyan",
            )
        )


if __name__ == "__main__":
    inspect_blockchain()
