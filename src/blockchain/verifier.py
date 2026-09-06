"""
Independent Proof Verification Module.
Queries on-chain smart contracts or transaction data to verify authenticity against local files.
"""

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from web3 import Web3

from ..config import NETWORKS, DEFAULT_NETWORK, REGISTRY_CONTRACT_ADDRESS
from .contract_abi import FACE_PROOF_REGISTRY_ABI
from ..utils.crypto_utils import compute_file_sha256, to_bytes32_hex, compute_merkle_root


@dataclass
class MatchVerificationOutcome:
    """Result of independent on-chain match record verification."""
    is_valid: bool
    status: str
    network: str
    contract_address: str
    tx_hash: Optional[str]
    block_number: Optional[int]
    local_match_record_hash: str
    on_chain_match_record_hash: str
    hash_matched: bool
    local_candidate_content_hash: str
    on_chain_candidate_content_hash: str
    content_hash_matched: bool
    local_url: str
    on_chain_url: str
    url_matched: bool
    platform: str
    content_type: str
    similarity_score: str
    timestamp: int
    raw_on_chain_record: Dict[str, Any]
    tamper_detected: bool = False
    tamper_details: str = ""


@dataclass
class VerificationOutcome:
    """Result of an independent blockchain proof verification."""
    is_valid: bool
    status: str
    network: str
    tx_hash: Optional[str]
    block_number: Optional[int]
    image_hash_matched: bool
    on_chain_image_hash: str
    local_image_hash: str
    on_chain_face_vector_hash: str
    on_chain_matched_image_hash: str
    social_post_url: str
    similarity_score: str
    contract_address: str
    timestamp: int
    raw_record: Dict[str, Any]


class ProofVerifier:
    """
    Independent verifier that queries blockchain records to validate match proof integrity.
    Verifies discovered match record against on-chain smart contract state.
    """

    def __init__(self, network_key: str = DEFAULT_NETWORK, contract_address: Optional[str] = None):
        self.network_key = network_key if network_key in NETWORKS else "polygon_amoy"
        self.net_config = NETWORKS[self.network_key]
        self.w3 = self._init_web3_connection()
        self.contract_address = contract_address if (contract_address is not None and contract_address.strip() != "") else (os.getenv("REGISTRY_CONTRACT_ADDRESS", "") or REGISTRY_CONTRACT_ADDRESS)

    def _init_web3_connection(self) -> Web3:
        """Initializes Web3 connection with fallback RPC support."""
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

    def verify_match_record(
        self,
        match_record_or_receipt: Any,
        contract_address: Optional[str] = None,
        receipt_tx_hash: Optional[str] = None,
        receipt_block_number: Optional[int] = None,
    ) -> MatchVerificationOutcome:
        """
        Independent On-Chain Re-Verification of a discovered social media match.
        1. Loads/extracts match record data.
        2. Deterministically reconstructs the CanonicalMatchRecord.
        3. Recalculates local matchRecordHash.
        4. Queries the smart contract on-chain state for matchRecordHash.
        5. Compares local hashes & fields against on-chain immutable records.
        6. Flags tampering if any field has been modified.
        """
        from ..matcher.match_record import CanonicalMatchRecord, normalize_url

        target_contract = contract_address or self.contract_address
        receipt_hash = ""
        tx_hash = receipt_tx_hash
        block_num = receipt_block_number

        # Extract dictionary from receipt file path or dict
        if isinstance(match_record_or_receipt, str) and os.path.exists(match_record_or_receipt):
            with open(match_record_or_receipt, "r", encoding="utf-8") as f:
                data = json.load(f)
            match_dict = data.get("discovered_match", data)
            b_proof = data.get("blockchain_proof") or data.get("blockchain", {})
            receipt_hash = b_proof.get("match_record_hash") or b_proof.get("image_hash", "")
            target_contract = target_contract or b_proof.get("contract_address", "")
            tx_hash = tx_hash or b_proof.get("transaction_hash") or b_proof.get("tx_hash")
            block_num = block_num or b_proof.get("block_number")
        elif isinstance(match_record_or_receipt, dict):
            if "discovered_match" in match_record_or_receipt:
                match_dict = match_record_or_receipt["discovered_match"]
                b_proof = match_record_or_receipt.get("blockchain_proof") or match_record_or_receipt.get("blockchain", {})
                receipt_hash = b_proof.get("match_record_hash") or b_proof.get("image_hash", "")
                target_contract = target_contract or b_proof.get("contract_address", "")
                tx_hash = tx_hash or b_proof.get("transaction_hash") or b_proof.get("tx_hash")
                block_num = block_num or b_proof.get("block_number")
            else:
                match_dict = match_record_or_receipt
                b_proof = match_record_or_receipt.get("blockchain", {})
                receipt_hash = b_proof.get("match_record_hash") or b_proof.get("image_hash", "")
                target_contract = target_contract or b_proof.get("contract_address", "")
                tx_hash = tx_hash or b_proof.get("transaction_hash") or b_proof.get("tx_hash")
                block_num = block_num or b_proof.get("block_number")
        elif hasattr(match_record_or_receipt, "to_dict"):
            match_dict = match_record_or_receipt.to_dict()
        else:
            match_dict = dict(match_record_or_receipt)

        # Build canonical match record
        rec = CanonicalMatchRecord(
            platform=str(match_dict.get("platform", "")),
            content_type=str(match_dict.get("content_type", "Post")),
            post_url=str(match_dict.get("post_url") or match_dict.get("matched_social_url", "")),
            candidate_content_hash=str(match_dict.get("candidate_content_hash") or match_dict.get("candidate_image_hash", "")),
            face_similarity=float(match_dict.get("face_similarity") or match_dict.get("similarity_score", 0.0)),
            face_verification=str(match_dict.get("face_verification", "VERIFIED")),
            search_engine=str(match_dict.get("search_engine", "Reverse Image Search")),
            discovery_timestamp=int(match_dict.get("discovery_timestamp") or match_dict.get("timestamp", 0)),
        )

        local_match_record_hash = rec.compute_match_record_hash()
        local_cand_hash = rec.candidate_content_hash
        local_url = rec.post_url

        # Upfront check for receipt-level tampering
        if receipt_hash and receipt_hash.lower() != local_match_record_hash.lower():
            return MatchVerificationOutcome(
                is_valid=False,
                status="❌ MATCH RECORD VERIFICATION FAILED\n❌ ON-CHAIN RECORD DOES NOT MATCH CURRENT DATA\n⚠️ POSSIBLE TAMPERING / DATA MODIFICATION DETECTED",
                network=self.net_config["name"],
                contract_address=target_contract,
                tx_hash=tx_hash,
                block_number=block_num,
                local_match_record_hash=local_match_record_hash,
                on_chain_match_record_hash=receipt_hash,
                hash_matched=False,
                local_candidate_content_hash=local_cand_hash,
                on_chain_candidate_content_hash="",
                content_hash_matched=False,
                local_url=local_url,
                on_chain_url="",
                url_matched=False,
                platform=rec.platform,
                content_type=rec.content_type,
                similarity_score=f"{rec.face_similarity * 100:.2f}%",
                timestamp=0,
                raw_on_chain_record={},
                tamper_detected=True,
                tamper_details=f"Recalculated match record hash ({local_match_record_hash}) differs from registered proof hash ({receipt_hash}). Modifying match fields breaks cryptographic integrity.",
            )

        if not target_contract or not self.w3.is_address(target_contract):
            return MatchVerificationOutcome(
                is_valid=False,
                status="ERROR: Invalid or missing contract address",
                network=self.net_config["name"],
                contract_address=str(target_contract),
                tx_hash=tx_hash,
                block_number=block_num,
                local_match_record_hash=local_match_record_hash,
                on_chain_match_record_hash="",
                hash_matched=False,
                local_candidate_content_hash=local_cand_hash,
                on_chain_candidate_content_hash="",
                content_hash_matched=False,
                local_url=local_url,
                on_chain_url="",
                url_matched=False,
                platform=rec.platform,
                content_type=rec.content_type,
                similarity_score=f"{rec.face_similarity * 100:.2f}%",
                timestamp=0,
                raw_on_chain_record={},
                tamper_detected=True,
                tamper_details="Invalid contract address for on-chain query.",
            )

        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(target_contract),
            abi=FACE_PROOF_REGISTRY_ABI,
        )

        # Query on-chain smart contract using local match record hash
        lookup_hash = local_match_record_hash
        try:
            res = contract.functions.getProof(
                bytes.fromhex(lookup_hash.replace("0x", ""))
            ).call()
            exists, on_input_hash, on_cand_hash, on_url, sim_score, on_meta, ts, verifier = res

            if not exists:
                return MatchVerificationOutcome(
                    is_valid=False,
                    status="NOT FOUND: No proof record found on-chain for this matchRecordHash",
                    network=self.net_config["name"],
                    contract_address=target_contract,
                    tx_hash=tx_hash,
                    block_number=block_num,
                    local_match_record_hash=local_match_record_hash,
                    on_chain_match_record_hash="",
                    hash_matched=False,
                    local_candidate_content_hash=local_cand_hash,
                    on_chain_candidate_content_hash="",
                    content_hash_matched=False,
                    local_url=local_url,
                    on_chain_url="",
                    url_matched=False,
                    platform=rec.platform,
                    content_type=rec.content_type,
                    similarity_score=f"{rec.face_similarity * 100:.2f}%",
                    timestamp=0,
                    raw_on_chain_record={},
                    tamper_detected=False,
                    tamper_details="Record not found on blockchain.",
                )

            on_cand_hex = on_cand_hash.hex()
            hash_matched = True
            content_hash_matched = (local_cand_hash.lower().replace("0x", "") == on_cand_hex.lower().replace("0x", ""))
            url_matched = (normalize_url(local_url) == normalize_url(on_url))
            score_matched = abs(int(round(rec.face_similarity * 10000)) - sim_score) <= 50

            all_matched = hash_matched and content_hash_matched and url_matched and score_matched

            return MatchVerificationOutcome(
                is_valid=all_matched,
                status="VALID: Match record verified against immutable on-chain record" if all_matched else "MISMATCH: On-chain fields differ from local match record",
                network=self.net_config["name"],
                contract_address=target_contract,
                tx_hash=tx_hash,
                block_number=block_num,
                local_match_record_hash=local_match_record_hash,
                on_chain_match_record_hash=local_match_record_hash,
                hash_matched=hash_matched,
                local_candidate_content_hash=local_cand_hash,
                on_chain_candidate_content_hash=on_cand_hex,
                content_hash_matched=content_hash_matched,
                local_url=local_url,
                on_chain_url=on_url,
                url_matched=url_matched,
                platform=rec.platform,
                content_type=rec.content_type,
                similarity_score=f"{sim_score / 100.0:.2f}%",
                timestamp=ts,
                raw_on_chain_record={"verifier": verifier, "metadata_json": on_meta},
                tamper_detected=not all_matched,
                tamper_details="" if all_matched else "Discrepancy detected between match record and on-chain values.",
            )

        except Exception as e:
            return MatchVerificationOutcome(
                is_valid=False,
                status=f"QUERY ERROR: {str(e)}",
                network=self.net_config["name"],
                contract_address=target_contract,
                tx_hash=tx_hash,
                block_number=block_num,
                local_match_record_hash=local_match_record_hash,
                on_chain_match_record_hash="",
                hash_matched=False,
                local_candidate_content_hash=local_cand_hash,
                on_chain_candidate_content_hash="",
                content_hash_matched=False,
                local_url=local_url,
                on_chain_url="",
                url_matched=False,
                platform=rec.platform,
                content_type=rec.content_type,
                similarity_score=f"{rec.face_similarity * 100:.2f}%",
                timestamp=0,
                raw_on_chain_record={},
                tamper_detected=False,
                tamper_details=str(e),
            )

    def verify_image_by_contract(
        self, image_path: str, contract_address: Optional[str] = None
    ) -> VerificationOutcome:
        """
        Looks up an image's proof directly from the smart contract using its SHA-256 digest.
        Validates all on-chain fields against local computation.
        """
        local_hash = compute_file_sha256(image_path)
        local_bytes32 = to_bytes32_hex(local_hash)

        target_contract = contract_address or self.contract_address
        if not target_contract or not self.w3.is_address(target_contract):
            return VerificationOutcome(
                is_valid=False,
                status="ERROR: Invalid contract address",
                network=self.net_config["name"],
                tx_hash=None,
                block_number=None,
                image_hash_matched=False,
                on_chain_image_hash="",
                local_image_hash=local_bytes32,
                on_chain_face_vector_hash="",
                on_chain_matched_image_hash="",
                social_post_url="",
                similarity_score="0%",
                contract_address=str(target_contract),
                timestamp=0,
                raw_record={},
            )

        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(target_contract),
            abi=FACE_PROOF_REGISTRY_ABI,
        )

        try:
            res = contract.functions.getProof(
                bytes.fromhex(local_bytes32.replace("0x", ""))
            ).call()
            exists, face_vec, matched_img, social_url, sim_score, meta_str, ts, verifier = res

            if not exists:
                return VerificationOutcome(
                    is_valid=False,
                    status="NOT FOUND: No proof registered for this image hash on-chain",
                    network=self.net_config["name"],
                    tx_hash=None,
                    block_number=None,
                    image_hash_matched=False,
                    on_chain_image_hash="",
                    local_image_hash=local_bytes32,
                    on_chain_face_vector_hash="",
                    on_chain_matched_image_hash="",
                    social_post_url="",
                    similarity_score="0%",
                    contract_address=target_contract,
                    timestamp=0,
                    raw_record={},
                )

            face_vec_hex = "0x" + face_vec.hex()
            matched_img_hex = "0x" + matched_img.hex()
            is_valid_hashes = (face_vec != bytes(32)) and (matched_img != bytes(32))
            is_valid_url = len(social_url.strip()) > 0
            is_valid_score = (0 <= sim_score <= 10000)

            all_valid = is_valid_hashes and is_valid_url and is_valid_score

            return VerificationOutcome(
                is_valid=all_valid,
                status="VALID: Image hash matched immutable smart-contract record on-chain" if all_valid else "INVALID: Corrupted on-chain record fields",
                network=self.net_config["name"],
                tx_hash=None,
                block_number=None,
                image_hash_matched=True,
                on_chain_image_hash=local_bytes32,
                local_image_hash=local_bytes32,
                on_chain_face_vector_hash=face_vec_hex,
                on_chain_matched_image_hash=matched_img_hex,
                social_post_url=social_url,
                similarity_score=f"{sim_score / 100.0:.2f}%",
                contract_address=target_contract,
                timestamp=ts,
                raw_record={"metadata": meta_str, "verifier": verifier},
            )
        except Exception as e:
            return VerificationOutcome(
                is_valid=False,
                status=f"ERROR: Contract call failed ({str(e)})",
                network=self.net_config["name"],
                tx_hash=None,
                block_number=None,
                image_hash_matched=False,
                on_chain_image_hash="",
                local_image_hash=local_bytes32,
                on_chain_face_vector_hash="",
                on_chain_matched_image_hash="",
                social_post_url="",
                similarity_score="0%",
                contract_address=target_contract,
                timestamp=0,
                raw_record={},
            )

    def verify_by_tx_hash(
        self, tx_hash: str, image_path: Optional[str] = None
    ) -> VerificationOutcome:
        """Parses transaction calldata to verify proof integrity."""
        local_hash = compute_file_sha256(image_path) if image_path else ""
        local_bytes32 = to_bytes32_hex(local_hash) if local_hash else ""

        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            input_data = tx.get("input", "")
            if isinstance(input_data, bytes):
                input_data = input_data.hex()
            
            if input_data.startswith("0x"):
                raw_bytes = bytes.fromhex(input_data[2:])
            else:
                raw_bytes = bytes.fromhex(input_data)

            payload_json = json.loads(raw_bytes.decode("utf-8"))
            on_chain_img_hash = payload_json.get("image_hash", "")
            
            hash_matched = (local_bytes32 == on_chain_img_hash) if local_bytes32 else True

            return VerificationOutcome(
                is_valid=hash_matched,
                status="VALID: On-chain transaction payload verified" if hash_matched else "MISMATCH: Image hash does not match tx payload",
                network=self.net_config["name"],
                tx_hash=tx_hash,
                block_number=tx.get("blockNumber"),
                image_hash_matched=hash_matched,
                on_chain_image_hash=on_chain_img_hash,
                local_image_hash=local_bytes32,
                on_chain_face_vector_hash=payload_json.get("face_vector_hash", ""),
                on_chain_matched_image_hash=payload_json.get("matched_image_hash", ""),
                social_post_url=payload_json.get("social_post_url", ""),
                similarity_score=f"{payload_json.get('similarity_score', 0) / 100.0:.2f}%",
                contract_address="",
                timestamp=payload_json.get("timestamp", 0),
                raw_record=payload_json,
            )
        except Exception as e:
            return VerificationOutcome(
                is_valid=False,
                status=f"ERROR: Failed to retrieve or parse transaction ({str(e)})",
                network=self.net_config["name"],
                tx_hash=tx_hash,
                block_number=None,
                image_hash_matched=False,
                on_chain_image_hash="",
                local_image_hash=local_bytes32,
                on_chain_face_vector_hash="",
                on_chain_matched_image_hash="",
                social_post_url="",
                similarity_score="0%",
                contract_address="",
                timestamp=0,
                raw_record={},
            )

    def verify_by_receipt_file(
        self, receipt_file_path: str, image_path: Optional[str] = None
    ) -> VerificationOutcome:
        """Verifies local image integrity against a blockchain receipt file."""
        local_hash = compute_file_sha256(image_path) if image_path else ""
        local_bytes32 = to_bytes32_hex(local_hash) if local_hash else ""

        try:
            with open(receipt_file_path, "r") as f:
                data = json.load(f)

            recorded_img_hash = data.get("image_sha256", "")
            recorded_bytes32 = to_bytes32_hex(recorded_img_hash) if recorded_img_hash else ""
            
            is_matched = (local_hash.lower() == recorded_img_hash.lower()) if local_hash else True
            bc = data.get("blockchain", {})

            return VerificationOutcome(
                is_valid=is_matched,
                status="VALID: Local image SHA-256 matches blockchain proof receipt" if is_matched else "MISMATCH: Local image SHA-256 differs from registered proof",
                network=bc.get("network", self.net_config["name"]),
                tx_hash=bc.get("tx_hash"),
                block_number=bc.get("block_number"),
                image_hash_matched=is_matched,
                on_chain_image_hash=recorded_bytes32,
                local_image_hash=local_bytes32,
                on_chain_face_vector_hash=data.get("face_vector_hash", ""),
                on_chain_matched_image_hash=data.get("matched_image_hash", ""),
                social_post_url=data.get("matched_social_url", ""),
                similarity_score=f"{data.get('similarity_score', 0) * 100.0:.2f}%",
                contract_address=bc.get("contract_address", ""),
                timestamp=data.get("timestamp", 0),
                raw_record=data,
            )
        except Exception as e:
            return VerificationOutcome(
                is_valid=False,
                status=f"ERROR: Failed to read proof receipt ({str(e)})",
                network=self.net_config["name"],
                tx_hash=None,
                block_number=None,
                image_hash_matched=False,
                on_chain_image_hash="",
                local_image_hash=local_bytes32,
                on_chain_face_vector_hash="",
                on_chain_matched_image_hash="",
                social_post_url="",
                similarity_score="0%",
                contract_address="",
                timestamp=0,
                raw_record={},
            )
