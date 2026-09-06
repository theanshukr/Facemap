"""
Blockchain Notary Agent.
Uploads tamper-evident face verification proofs to EVM blockchain networks (Polygon Amoy / Ethereum Sepolia).
Enforces strict smart contract execution on Polygon Amoy for production and evaluator verification.
Zero tolerance for simulation or offline fallbacks in live-demo mode.
"""

import json
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from web3 import Web3
from eth_account import Account

from ..config import (
    NETWORKS,
    DEFAULT_NETWORK,
    PRIVATE_KEY,
    REGISTRY_CONTRACT_ADDRESS,
)
from .contract_abi import FACE_PROOF_REGISTRY_ABI
from ..utils.crypto_utils import to_bytes32_hex, compute_merkle_root


@dataclass
class BlockchainReceipt:
    """Receipt returned after anchoring verified match proof on-chain."""
    match_record_hash: str
    candidate_content_hash: str
    input_image_hash: str
    tx_hash: str
    block_number: int
    network_name: str
    chain_id: int
    contract_address: str
    explorer_url: str
    gas_used: int
    wallet_address: str
    timestamp: int
    match_record: Dict[str, Any]
    payload_summary: Dict[str, Any]


class BlockchainNotary:
    """
    Connects to EVM networks and submits cryptographic match proof records.
    Enforces deployed smart contract verification on Polygon Amoy.
    Anchors deterministic matchRecordHash as primary on-chain identifier.
    """

    def __init__(
        self,
        network_key: str = DEFAULT_NETWORK,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        self.network_key = network_key if network_key in NETWORKS else "polygon_amoy"
        self.net_config = NETWORKS[self.network_key]
        self.chain_id = self.net_config["chain_id"]
        
        # Connect with automatic RPC failover
        self.w3 = self._init_web3_connection()

        # Wallet initialization
        self.raw_private_key = private_key if private_key is not None else (os.getenv("PRIVATE_KEY", "") or PRIVATE_KEY)
        if self.raw_private_key and self.raw_private_key.strip():
            pk = self.raw_private_key.strip()
            if not pk.startswith("0x"):
                pk = "0x" + pk
            try:
                self.account = Account.from_key(pk)
            except Exception:
                self.account = None
        else:
            self.account = None

        self.contract_address = contract_address if (contract_address is not None and contract_address.strip() != "") else (os.getenv("REGISTRY_CONTRACT_ADDRESS", "") or REGISTRY_CONTRACT_ADDRESS)
        self.contract = None
        if self.contract_address and self.w3.is_address(self.contract_address):
            checksum_addr = Web3.to_checksum_address(self.contract_address)
            self.contract = self.w3.eth.contract(
                address=checksum_addr,
                abi=FACE_PROOF_REGISTRY_ABI,
            )

    def _init_web3_connection(self) -> Web3:
        """Initializes Web3 connection, trying fallback RPCs if the primary is unreachable."""
        candidate_rpcs: List[str] = self.net_config.get("rpc_fallback_urls", [self.net_config["rpc_url"]])
        for rpc in candidate_rpcs:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                if w3.is_connected():
                    try:
                        from web3.middleware import ExtraDataToPOAMiddleware
                        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                    except Exception:
                        pass
                    return w3
            except Exception:
                continue
        w3 = Web3(Web3.HTTPProvider(self.net_config["rpc_url"], request_kwargs={"timeout": 10}))
        try:
            from web3.middleware import ExtraDataToPOAMiddleware
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except Exception:
            pass
        return w3

    @property
    def is_connected(self) -> bool:
        """Check if Web3 connection to RPC is live."""
        try:
            return bool(self.w3.is_connected())
        except Exception:
            return False

    def get_wallet_balance(self) -> float:
        """Returns wallet native balance in ETH/POL."""
        if not self.account:
            return 0.0
        try:
            bal_wei = self.w3.eth.get_balance(self.account.address)
            return float(self.w3.from_wei(bal_wei, "ether"))
        except Exception:
            return 0.0

    def verify_contract_deployed(self) -> bool:
        """Verifies that the target contract exists on-chain with non-empty bytecode."""
        if not self.contract_address or not self.w3.is_address(self.contract_address):
            return False
        try:
            code = self.w3.eth.get_code(Web3.to_checksum_address(self.contract_address))
            return len(code) > 2  # more than empty '0x'
        except Exception:
            return False

    def anchor_match_proof(
        self,
        match_record: Any,  # CanonicalMatchRecord or Dict
        input_image_sha256: str,
        require_live_contract: bool = False,
    ) -> BlockchainReceipt:
        """
        Submits the verified social-media match record to the blockchain registry.
        Uses matchRecordHash as the primary on-chain lookup key,
        candidateContentHash as the discovered artifact fingerprint,
        and inputImageHash strictly as provenance.
        """
        # Determine match_record details
        if hasattr(match_record, "to_dict"):
            match_dict = match_record.to_dict()
            canonical_json = match_record.to_canonical_json()
            match_record_hash = match_record.compute_match_record_hash()
            candidate_content_hash = match_record.candidate_content_hash
            post_url = match_record.post_url
            sim_score = match_record.face_similarity
        else:
            from ..matcher.match_record import CanonicalMatchRecord, normalize_url
            match_dict = dict(match_record)
            rec = CanonicalMatchRecord(
                platform=match_dict.get("platform", ""),
                content_type=match_dict.get("content_type", ""),
                post_url=match_dict.get("post_url", ""),
                candidate_content_hash=match_dict.get("candidate_content_hash", ""),
                face_similarity=float(match_dict.get("face_similarity", 0.0)),
                face_verification=match_dict.get("face_verification", "VERIFIED"),
                search_engine=match_dict.get("search_engine", "Reverse Image Search"),
                discovery_timestamp=int(match_dict.get("discovery_timestamp", 0)),
            )
            canonical_json = rec.to_canonical_json()
            match_record_hash = rec.compute_match_record_hash()
            candidate_content_hash = rec.candidate_content_hash
            post_url = rec.post_url
            sim_score = rec.face_similarity

        match_record_bytes32 = to_bytes32_hex(match_record_hash)
        candidate_content_bytes32 = to_bytes32_hex(candidate_content_hash)
        input_image_bytes32 = to_bytes32_hex(input_image_sha256)
        similarity_score_scaled = int(round(sim_score * 10000))

        if require_live_contract:
            # 1. Check RPC connection
            if not self.is_connected:
                raise RuntimeError(
                    f"Cannot connect to RPC endpoint for {self.net_config['name']} ({self.net_config['rpc_url']}). "
                    "Please check your network connection."
                )

            # 2. Check Private Key
            if not self.account:
                raise RuntimeError(
                    "PRIVATE_KEY is missing or invalid in .env. "
                    "A valid private key is strictly required for live blockchain notarization."
                )

            # 3. Check Deployed Contract Address & Bytecode
            if not self.contract_address or not self.w3.is_address(self.contract_address):
                raise RuntimeError(
                    "REGISTRY_CONTRACT_ADDRESS is not set or not a valid EVM address in .env. "
                    "Please set a valid deployed FaceProofRegistry contract address."
                )

            if not self.verify_contract_deployed():
                raise RuntimeError(
                    f"No deployed smart contract code found at address {self.contract_address} on {self.net_config['name']}. "
                    "Please deploy FaceProofRegistry.sol (using deploy_contract.py) and update REGISTRY_CONTRACT_ADDRESS."
                )

            # 4. Check Wallet Balance
            balance = self.get_wallet_balance()
            if balance <= 0:
                raise RuntimeError(
                    f"Wallet {self.account.address} has 0.0 native balance on {self.net_config['name']}. "
                    "Please fund this wallet with testnet tokens (POL/MATIC) from the faucet to send transactions."
                )

            # 5. Submit real contract transaction
            return self._anchor_via_contract(
                match_record_bytes32=match_record_bytes32,
                input_image_bytes32=input_image_bytes32,
                candidate_content_bytes32=candidate_content_bytes32,
                post_url=post_url,
                similarity_score_scaled=similarity_score_scaled,
                canonical_json=canonical_json,
                match_record_hash=match_record_hash,
                candidate_content_hash=candidate_content_hash,
                input_image_sha256=input_image_sha256,
                match_dict=match_dict,
            )

        # Developer / Non-strict branch
        if self.contract and self.is_connected and self.verify_contract_deployed() and self.account and self.get_wallet_balance() > 0:
            return self._anchor_via_contract(
                match_record_bytes32=match_record_bytes32,
                input_image_bytes32=input_image_bytes32,
                candidate_content_bytes32=candidate_content_bytes32,
                post_url=post_url,
                similarity_score_scaled=similarity_score_scaled,
                canonical_json=canonical_json,
                match_record_hash=match_record_hash,
                candidate_content_hash=candidate_content_hash,
                input_image_sha256=input_image_sha256,
                match_dict=match_dict,
            )
        else:
            raise RuntimeError(
                "Blockchain notarization requires a deployed contract and funded wallet. "
                "Simulated offline notarization has been permanently disabled."
            )

    def anchor_proof(
        self,
        image_sha256: str,
        face_vector_hash: str,
        matched_image_hash: str,
        social_post_url: str,
        similarity_score_scaled: int,
        metadata: Dict[str, Any],
        require_live_contract: bool = False,
    ) -> BlockchainReceipt:
        """Backwards-compatible wrapper that constructs a CanonicalMatchRecord and anchors it."""
        from ..matcher.match_record import build_match_record
        rec = build_match_record(
            platform=metadata.get("social_platform", "Social"),
            content_type=metadata.get("content_type", "Post"),
            post_url=social_post_url,
            candidate_content_hash=matched_image_hash,
            face_similarity=metadata.get("similarity_score", similarity_score_scaled / 10000.0),
            face_verification="VERIFIED",
            search_engine=metadata.get("search_source", "Reverse Image Search"),
        )
        return self.anchor_match_proof(
            match_record=rec,
            input_image_sha256=image_sha256,
            require_live_contract=require_live_contract,
        )

    def _anchor_via_contract(
        self,
        match_record_bytes32: str,
        input_image_bytes32: str,
        candidate_content_bytes32: str,
        post_url: str,
        similarity_score_scaled: int,
        canonical_json: str,
        match_record_hash: str,
        candidate_content_hash: str,
        input_image_sha256: str,
        match_dict: Dict[str, Any],
    ) -> BlockchainReceipt:
        """Builds, signs, and executes the smart-contract registerProof transaction on Polygon Amoy."""
        checksum_contract_addr = Web3.to_checksum_address(self.contract_address)
        contract = self.w3.eth.contract(
            address=checksum_contract_addr,
            abi=FACE_PROOF_REGISTRY_ABI,
        )

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        gas_price = self.w3.eth.gas_price

        # Primary on-chain key: match_record_hash
        arg_match_hash = bytes.fromhex(match_record_bytes32.replace("0x", ""))
        arg_input_hash = bytes.fromhex(input_image_bytes32.replace("0x", ""))
        arg_cand_hash = bytes.fromhex(candidate_content_bytes32.replace("0x", ""))

        tx_func = contract.functions.registerProof(
            arg_match_hash,
            arg_input_hash,
            arg_cand_hash,
            post_url,
            similarity_score_scaled,
            canonical_json,
        )

        # Dynamic Gas Estimation & Pre-Flight Cost Verification
        try:
            gas_estimate = tx_func.estimate_gas({"from": self.account.address})
            gas_limit = int(gas_estimate * 1.2)
        except Exception as est_err:
            # Fallback if RPC estimate fails (standard proof registration consumes ~200k gas)
            gas_estimate = 220000
            gas_limit = 260000

        estimated_cost_wei = gas_limit * gas_price
        balance_wei = self.w3.eth.get_balance(self.account.address)

        # Safe Pre-Flight Diagnostics (No secrets exposed)
        from ..utils.ui_display import console
        console.print(f"[dim]• Network Chain ID: {self.chain_id} | Contract: {self.contract_address}[/dim]")
        console.print(f"[dim]• Gas Limit: {gas_limit:,} | Gas Price: {self.w3.from_wei(gas_price, 'gwei'):.2f} Gwei[/dim]")
        console.print(f"[dim]• Estimated Tx Cost: {self.w3.from_wei(estimated_cost_wei, 'ether'):.6f} POL | Wallet Balance: {self.w3.from_wei(balance_wei, 'ether'):.6f} POL[/dim]")

        if balance_wei < estimated_cost_wei:
            bal_pol = self.w3.from_wei(balance_wei, "ether")
            cost_pol = self.w3.from_wei(estimated_cost_wei, "ether")
            raise RuntimeError(
                f"Insufficient POL balance on Polygon Amoy for gas.\n"
                f"  - Wallet: {self.account.address}\n"
                f"  - Current Balance: {bal_pol} POL\n"
                f"  - Estimated Gas Cost: {cost_pol} POL ({gas_limit:,} gas @ {self.w3.from_wei(gas_price, 'gwei'):.2f} Gwei)\n"
                f"Please obtain testnet POL (e.g. from an Amoy faucet) to execute live blockchain anchoring."
            )

        # Build transaction
        tx_data = tx_func.build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "chainId": self.chain_id,
        })

        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx_data, private_key=self.account.key)

        # Broadcast transaction
        tx_hash_bytes = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = tx_hash_bytes.hex()
        if not tx_hash_hex.startswith("0x"):
            tx_hash_hex = "0x" + tx_hash_hex

        # Wait for mined receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=120)
        
        if receipt.status != 1:
            raise RuntimeError(f"Smart-contract transaction {tx_hash_hex} reverted on-chain (status 0).")

        explorer_url = f"{self.net_config['explorer_tx_url']}{tx_hash_hex}"

        return BlockchainReceipt(
            match_record_hash=match_record_hash,
            candidate_content_hash=candidate_content_hash,
            input_image_hash=input_image_sha256,
            tx_hash=tx_hash_hex,
            block_number=receipt.blockNumber,
            network_name=self.net_config["name"],
            chain_id=self.chain_id,
            contract_address=self.contract_address,
            explorer_url=explorer_url,
            gas_used=receipt.gasUsed,
            wallet_address=self.account.address,
            timestamp=int(time.time()),
            match_record=match_dict,
            payload_summary={
                "match_record_hash": match_record_hash,
                "candidate_content_hash": candidate_content_hash,
                "input_image_hash": input_image_sha256,
                "post_url": post_url,
                "similarity_score_scaled": similarity_score_scaled,
            },
        )
