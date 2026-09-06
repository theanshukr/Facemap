#!/usr/bin/env python3
"""
Deploy FaceProofRegistry Smart Contract to Polygon Amoy, Sepolia, or Local EVM.
Deploys real EVM bytecode, waits for mined receipt, verifies on-chain bytecode,
prints transaction details & explorer URL, and automatically updates .env.
"""

import os
import re
import sys
import argparse
from typing import Optional
from web3 import Web3
from eth_account import Account
from rich.console import Console
from rich.panel import Panel
from rich import box

from src.config import NETWORKS, DEFAULT_NETWORK, PRIVATE_KEY, BASE_DIR
from src.blockchain.contract_abi import FACE_PROOF_REGISTRY_ABI, FACE_PROOF_REGISTRY_BYTECODE

console = Console()


def update_env_contract_address(contract_address: str, env_path: Optional[str] = None):
    """
    Updates or inserts REGISTRY_CONTRACT_ADDRESS in .env without modifying secrets.
    """
    target_env = env_path or os.path.join(BASE_DIR, ".env")
    if not os.path.exists(target_env):
        with open(target_env, "w", encoding="utf-8") as f:
            f.write(f"REGISTRY_CONTRACT_ADDRESS={contract_address}\n")
        return

    with open(target_env, "r", encoding="utf-8") as f:
        content = f.read()

    if re.search(r"^REGISTRY_CONTRACT_ADDRESS=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(
            r"^REGISTRY_CONTRACT_ADDRESS=.*$",
            f"REGISTRY_CONTRACT_ADDRESS={contract_address}",
            content,
            flags=re.MULTILINE,
        )
    else:
        new_content = content.rstrip() + f"\nREGISTRY_CONTRACT_ADDRESS={contract_address}\n"

    with open(target_env, "w", encoding="utf-8") as f:
        f.write(new_content)


def deploy(network_key: str = DEFAULT_NETWORK, private_key: str = ""):
    net_config = NETWORKS.get(network_key, NETWORKS["polygon_amoy"])
    candidate_rpcs = net_config.get("rpc_fallback_urls", [net_config["rpc_url"]])
    if net_config["rpc_url"] not in candidate_rpcs:
        candidate_rpcs.insert(0, net_config["rpc_url"])

    w3 = None
    active_rpc = None
    for rpc in candidate_rpcs:
        try:
            temp_w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 12}))
            if temp_w3.is_connected():
                w3 = temp_w3
                active_rpc = rpc
                break
        except Exception:
            continue

    if not w3 or not w3.is_connected():
        console.print(f"[bold red]Cannot connect to {net_config['name']} RPC at {candidate_rpcs}[/bold red]")
        sys.exit(1)

    raw_pk = private_key or PRIVATE_KEY
    if not raw_pk or not raw_pk.strip():
        console.print("[bold red]Please provide a private key via --private-key or in .env[/bold red]")
        sys.exit(1)

    pk = raw_pk.strip()
    if not pk.startswith("0x"):
        pk = "0x" + pk

    try:
        account = Account.from_key(pk)
    except Exception as e:
        console.print(f"[bold red]Invalid private key format: {e}[/bold red]")
        sys.exit(1)

    balance_wei = w3.eth.get_balance(account.address)
    balance_eth = w3.from_wei(balance_wei, "ether")

    console.print(f"Deployer address: [cyan]{account.address}[/cyan]")
    console.print(f"Network: [yellow]{net_config['name']}[/yellow] (Chain ID: {net_config['chain_id']})")
    console.print(f"Connected RPC: [dim]{active_rpc}[/dim]")
    console.print(f"Native Balance: [green]{balance_eth} ETH/POL[/green]\n")

    if balance_wei == 0:
        console.print(f"[bold red]Deployer wallet {account.address} has 0.0 POL balance. Please fund it via testnet faucet.[/bold red]")
        sys.exit(1)

    try:
        from web3.middleware import ExtraDataToPOAMiddleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except Exception:
        pass

    console.print("[bold cyan]Deploying FaceProofRegistry...[/bold cyan]")

    registry_factory = w3.eth.contract(
        abi=FACE_PROOF_REGISTRY_ABI,
        bytecode=FACE_PROOF_REGISTRY_BYTECODE,
    )

    nonce = w3.eth.get_transaction_count(account.address)
    gas_price = w3.eth.gas_price

    construct_tx = registry_factory.constructor().build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 600000,
        "gasPrice": gas_price,
        "chainId": net_config["chain_id"],
    })

    # Sign transaction
    signed_tx = w3.eth.account.sign_transaction(construct_tx, private_key=account.key)
    tx_hash_bytes = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    tx_hash = "0x" + tx_hash_bytes.hex()
    console.print(f"Deployment transaction broadcast: [magenta]{tx_hash}[/magenta]")
    console.print("[bold green]Waiting for transaction confirmation on Polygon Amoy...[/bold green]")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=180)

    if receipt.get("status") != 1:
        console.print(f"[bold red]Deployment failed on-chain with status 0. Tx: {tx_hash}[/bold red]")
        sys.exit(1)

    contract_address = receipt.get("contractAddress")
    if not contract_address or not w3.is_address(contract_address):
        console.print("[bold red]Error: No valid contract address returned in deployment receipt.[/bold red]")
        sys.exit(1)

    contract_address = Web3.to_checksum_address(contract_address)
    block_number = receipt.get("blockNumber")
    gas_used = receipt.get("gasUsed")

    # Verify on-chain bytecode
    code = w3.eth.get_code(contract_address)
    if len(code) <= 2:
        console.print(f"[bold red]Verification FAILED: Address {contract_address} has empty bytecode (not a valid contract).[/bold red]")
        sys.exit(1)

    # Automatically update .env
    update_env_contract_address(contract_address)

    explorer_url = f"{net_config['explorer_tx_url']}{tx_hash}"
    contract_explorer_url = f"https://amoy.polygonscan.com/address/{contract_address}" if net_config["chain_id"] == 80002 else f"{net_config['explorer_tx_url']}{contract_address}"

    console.print(
        Panel(
            f"[bold green]Deployment completed successfully![/bold green]\n\n"
            f"[cyan]Contract Address:[/cyan] [bold white]{contract_address}[/bold white]\n"
            f"[cyan]Transaction Hash:[/cyan] [magenta]{tx_hash}[/magenta]\n"
            f"[cyan]Block Number:[/cyan] {block_number}\n"
            f"[cyan]Gas Used:[/cyan] {gas_used:,}\n"
            f"[cyan]Contract Verification:[/cyan] [bold green]PASSED[/bold green]\n"
            f"[cyan]Bytecode Detected:[/cyan] [bold green]YES[/bold green] ({len(code)} bytes)\n"
            f"[cyan]PolygonScan Explorer:[/cyan] [underline blue]{contract_explorer_url}[/underline blue]\n"
            f"[cyan]Environment Configuration:[/cyan] [green]REGISTRY_CONTRACT_ADDRESS updated in .env[/green]",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    return contract_address, tx_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy FaceProofRegistry Contract")
    parser.add_argument("--network", "-n", default=DEFAULT_NETWORK, choices=["polygon_amoy", "sepolia", "local"])
    parser.add_argument("--private-key", "-k", default="")
    args = parser.parse_args()
    deploy(args.network, args.private_key)
