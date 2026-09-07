#!/usr/bin/env python3
"""
Face ID + Blockchain Verification Pipeline - Main Entrypoint.
Detects & encodes faces using deep neural models (YuNet + SFace),
searches genuine social media matches via live reverse-image search,
biometrically ranks and cross-verifies all candidate photos,
anchors immutable proof to Polygon Amoy smart contract (FaceProofRegistry),
and independently re-verifies proof directly from the blockchain on-chain state.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.face_engine.detector import FaceEngine
from src.search_engine.reverse_search import ReverseImageSearchEngine, SocialMatchCandidate
from src.search_engine.identity_resolver import IdentityResolver
from src.matcher.face_matcher import FaceMatcher, MatchEvaluationResult
from src.matcher.match_record import CanonicalMatchRecord, build_match_record
from src.blockchain.notary import BlockchainNotary
from src.blockchain.verifier import ProofVerifier
from src.utils.ui_display import (
    print_banner,
    print_step_header,
    print_face_detection_summary,
    print_search_results,
    print_candidate_ranking_table,
    print_match_analysis,
    print_match_evaluation,
    print_related_matches_table,
    print_step4_match_anchor,
    print_step5_match_verification,
    print_blockchain_receipt,
    print_final_verification_section,
    console,
)
from src.config import BASE_DIR, DEFAULT_NETWORK, SIMILARITY_THRESHOLD


def run_pipeline(
    image_path: str,
    network: str = DEFAULT_NETWORK,
    contract_address: str = "",
    threshold: float = SIMILARITY_THRESHOLD,
    save_receipt: str = "",
    direct_social_url: str = "",
    live_demo: bool = False,
    simulate_tamper: bool = False,
    skip_blockchain: bool = False,
):
    print_banner()

    if not save_receipt:
        img_stem = Path(image_path).stem if image_path else "match"
        receipts_dir = BASE_DIR / "receipts"
        receipts_dir.mkdir(exist_ok=True)
        save_receipt = str(receipts_dir / f"{img_stem}_receipt.json")

    if not os.path.exists(image_path):
        console.print(f"[bold red]Error: Input image '{image_path}' not found.[/bold red]")
        sys.exit(1)

    if live_demo and direct_social_url:
        console.print("[bold red]Error: Direct URL injection (--url) is strictly prohibited in --live-demo mode.[/bold red]")
        sys.exit(1)

    # -------------------------------------------------------------
    # STEP 1: Detect and Encode Face using Deep Neural Engine
    # -------------------------------------------------------------
    print_step_header(1, "Deep Neural Face Detection & Embedding (YuNet + SFace)")
    with console.status("[bold green]Detecting faces and computing 128-D neural embeddings..."):
        face_engine = FaceEngine()
        analysis = face_engine.analyze_image(image_path)

    if not analysis.faces:
        console.print("[bold red]❌ No face detected in input photo. Zero-fallback policy active. Please provide a clear portrait photo.[/bold red]")
        sys.exit(1)

    print_face_detection_summary(analysis)
    primary_face = analysis.primary_face

    # -------------------------------------------------------------
    # STEP 2: Live Reverse-Image Search Discovery
    # -------------------------------------------------------------
    print_step_header(2, "Live Reverse-Image Search for Social Media Matches")
    search_engine = ReverseImageSearchEngine()

    with console.status("[bold green]Querying reverse visual search index..."):
        try:
            candidates = search_engine.search_by_image(
                image_path=image_path,
                require_live=live_demo,
            )
        except Exception as e:
            console.print(f"[bold red]❌ Search Provider Error: {e}[/bold red]")
            sys.exit(1)

    if direct_social_url and not live_demo:
        candidates.insert(
            0,
            SocialMatchCandidate(
                post_url=direct_social_url,
                platform="social",
                title="User Specified Profile",
                author="User",
                image_url=None,
                source_engine="Reverse Image Search",
                snippet="Direct social media profile link",
                raw_metadata={},
                is_social_media=True,
            ),
        )

    if not candidates:
        console.print("[bold red]❌ Search Error: No matching candidates could be found via reverse visual search.[/bold red]")
        console.print("[yellow]Please verify your internet connection and active search API keys in .env.[/yellow]")
        sys.exit(1)

    print_search_results(candidates)

    # -------------------------------------------------------------
    # STEP 3: Multi-Candidate Download & Biometric Ranking
    # -------------------------------------------------------------
    print_step_header(3, "Multi-Candidate Face Ranking & Cross-Verification")
    matcher = FaceMatcher(threshold=threshold)

    console.print("[bold cyan]Processing candidates & evaluating biometric similarity:[/bold cyan]")
    
    def log_diag(msg: str):
        if "SKIPPED" in msg:
            console.print(f"  [dim yellow]• {msg}[/dim yellow]")
        elif "Face detection" in msg:
            console.print(f"  [bold green]• {msg}[/bold green]")
        else:
            console.print(f"  [dim]• {msg}[/dim]")

    ranking_report = matcher.rank_candidates(
        original_face=primary_face,
        candidates=candidates,
        download_fn=search_engine.download_candidate_image,
        log_callback=log_diag,
    )

    print_candidate_ranking_table(ranking_report)

    if not ranking_report.ranked_candidates:
        console.print("\n[bold red]❌ Biometric Rejection: No candidate image with a detectable face was found for comparison.[/bold red]")
        console.print("[bold red]Pipeline stopped. Blockchain proof will NOT be registered.[/bold red]")
        sys.exit(1)

    identity_cand = None
    try:
        identity_cand = IdentityResolver.resolve_identity(candidates)
    except Exception:
        pass

    top_result, selection_reason = ranking_report.select_final_social_match(threshold=threshold)
    is_social_winner = top_result is not None

    if top_result is None:
        top_non_social, non_soc_reason = ranking_report.select_top_non_social_match(threshold=threshold)
        if top_non_social is not None:
            top_result = top_non_social
            selection_reason = non_soc_reason
        else:
            console.print(f"\n[bold red]❌ NO MATCH: {non_soc_reason}[/bold red]")
            console.print(f"[bold yellow]Selection Strategy: Verified Face Match (≥ {threshold * 100:.0f}%)[/bold yellow]")
            console.print("[bold red]Pipeline stopped: No eligible candidate passed biometric criteria.[/bold red]")
            sys.exit(1)

    selected_candidate = top_result.candidate
    match_result = top_result.evaluation

    discovered_attribution = None
    if identity_cand and identity_cand.is_confident:
        discovered_attribution = identity_cand.name
    elif selected_candidate.name_attribution and IdentityResolver.is_valid_person_name(selected_candidate.name_attribution):
        discovered_attribution = selected_candidate.name_attribution

    c_type = top_result.content_type or ("Post" if is_social_winner else "Web Page")
    url_st = top_result.url_validation.status.value if top_result.url_validation else "ACCESSIBLE"

    if is_social_winner:
        console.print("\n[bold cyan]━━━ [WINNER SELECTION] ━━━[/bold cyan]")
        console.print(f"[bold green]🏆 PRIMARY SOCIAL WINNER[/bold green]")
        console.print(f"Platform:   [bold yellow]{selected_candidate.platform.upper()}[/bold yellow]")
        console.print(f"Similarity: [bold green]{match_result.similarity_score * 100:.2f}%[/bold green]")
        console.print(f"URL:        [magenta]{selected_candidate.post_url}[/magenta]")
    else:
        console.print("\n[bold cyan]━━━ [TOP MATCHED NON-SOCIAL DISCOVERY] ━━━[/bold cyan]")
        console.print(f"[bold yellow]🌐 TOP NON-SOCIAL WEB MATCH[/bold yellow]")
        console.print(f"Platform:   [bold yellow]{selected_candidate.platform.upper()}[/bold yellow]")
        console.print(f"Similarity: [bold green]{match_result.similarity_score * 100:.2f}%[/bold green]")
        console.print(f"URL:        [magenta]{selected_candidate.post_url}[/magenta]")
        console.print(f"[dim yellow]ℹ Notice: No genuine social-media post met the criteria; displaying highest-similarity verified web discovery.[/dim yellow]")

    print_match_evaluation(
        match_result,
        selected_candidate.post_url,
        is_social_media=is_social_winner,
        platform=selected_candidate.platform,
        content_type=c_type,
        url_status=url_st,
        attribution=discovered_attribution,
        selection_reason=selection_reason,
    )

    related_verified_matches: list = []
    if is_social_winner:
        classified = ranking_report.get_classified_results(
            primary_winner=top_result,
            threshold=threshold,
        )
        related_verified_matches = classified.get("related_matches", [])
        print_related_matches_table(related_verified_matches)

    print_match_analysis(match_result, threshold=threshold)

    if not is_social_winner:
        console.print("\n[bold yellow]⚡ Blockchain notarization is strictly reserved for genuine social media proofs (Non-social web match displayed above).[/bold yellow]")
        if save_receipt:
            out_data = {
                "input_provenance": {
                    "input_image_path": image_path,
                    "input_image_hash": analysis.image_sha256,
                    "face_vector_hash": primary_face.vector_hash,
                },
                "discovered_match": {
                    "platform": selected_candidate.platform,
                    "content_type": c_type,
                    "post_url": selected_candidate.post_url,
                    "candidate_content_hash": match_result.matched_image_hash,
                    "face_similarity": match_result.similarity_score,
                    "face_verification": "VERIFIED",
                    "search_engine": selected_candidate.source_engine or "Reverse Image Search",
                },
            }
            with open(save_receipt, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
            console.print(f"[bold green][+] Match results saved to {save_receipt}[/bold green]")
        
        print_final_verification_section(
            is_social_winner=False,
            winner_or_top_web=top_result,
            identity_name=discovered_attribution,
            confidence_str=f"{identity_cand.confidence * 100:.0f}%" if identity_cand else None,
            skip_blockchain=True,
            skip_reason="No verified social-media match",
        )
        console.print("\n[bold green]>>> Pipeline completed successfully with top matched non-social discovery.[/bold green]")
        return

    # -------------------------------------------------------------
    # BUILD CANONICAL MATCH RECORD (Primary Winner Only)
    # -------------------------------------------------------------
    match_record = build_match_record(
        platform=selected_candidate.platform,
        content_type=c_type,
        post_url=selected_candidate.post_url,
        candidate_content_hash=match_result.matched_image_hash,
        face_similarity=match_result.similarity_score,
        face_verification="VERIFIED",
        search_engine=selected_candidate.source_engine or "Reverse Image Search",
        discovery_timestamp=int(time.time()),
    )

    if skip_blockchain:
        console.print("\n[bold yellow]⚡ [STEP 4 & 5 SKIPPED] Blockchain anchoring skipped (--no-blockchain).[/bold yellow]")
        if save_receipt:
            out_data = {
                "input_provenance": {
                    "input_image_path": image_path,
                    "input_image_hash": analysis.image_sha256,
                    "face_vector_hash": primary_face.vector_hash,
                },
                "discovered_match": match_record.to_dict(),
                "related_verified_content": [getattr(m, "candidate", m).post_url for m in related_verified_matches],
            }
            with open(save_receipt, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
            console.print(f"[bold green][+] Match results saved to {save_receipt}[/bold green]")
        
        print_final_verification_section(
            is_social_winner=True,
            winner_or_top_web=top_result,
            identity_name=discovered_attribution,
            confidence_str=f"{identity_cand.confidence * 100:.0f}%" if identity_cand else None,
            skip_blockchain=True,
            skip_reason="Blockchain anchoring skipped (--no-blockchain)",
        )
        console.print("\n[bold green]>>> Pipeline completed successfully through Step 3 (Biometric Face Matching).[/bold green]")
        return

    # -------------------------------------------------------------
    # STEP 4: Anchor Verified Match to Blockchain
    # -------------------------------------------------------------
    print_step_header(4, "Anchor Verified Match to Blockchain")

    with console.status(f"[bold green]Connecting to {network.upper()} and anchoring match proof to smart contract..."):
        try:
            notary = BlockchainNotary(network_key=network, contract_address=contract_address)
            receipt = notary.anchor_match_proof(
                match_record=match_record,
                input_image_sha256=analysis.image_sha256,
                require_live_contract=live_demo,
            )
        except Exception as e:
            console.print(f"\n[bold red]❌ Blockchain Notarization Error: {e}[/bold red]")
            sys.exit(1)

    print_step4_match_anchor(match_record, analysis.image_sha256, receipt)

    # -------------------------------------------------------------
    # STEP 5: Independent On-Chain Match Re-Verification
    # -------------------------------------------------------------
    print_step_header(5, "Independent On-Chain Match Re-Verification")
    with console.status("[bold green]Querying smart contract on-chain state to verify immutable match record..."):
        verifier = ProofVerifier(network_key=network, contract_address=receipt.contract_address)

        record_to_verify = match_record
        if simulate_tamper:
            console.print("\n[bold yellow]⚠️ SIMULATING TAMPERING ATTACK FOR DEMONSTRATION...[/bold yellow]")
            console.print("[dim]Adversary modifies face similarity score in local record from "
                          f"{match_record.face_similarity * 100:.2f}% to 99.99%[/dim]")
            tampered_dict = match_record.to_dict()
            tampered_dict["face_similarity"] = 0.9999
            record_to_verify = CanonicalMatchRecord(**tampered_dict)

        outcome = verifier.verify_match_record(
            match_record_or_receipt=record_to_verify,
            contract_address=receipt.contract_address,
            receipt_tx_hash=receipt.tx_hash,
            receipt_block_number=receipt.block_number,
        )

    print_step5_match_verification(outcome)

    if not outcome.is_valid and not simulate_tamper:
        console.print(f"[bold red]❌ On-Chain Verification Failed: {outcome.status}[/bold red]")
        sys.exit(1)

    # Save receipt JSON if requested
    if save_receipt:
        out_data = {
            "input_provenance": {
                "input_image_path": image_path,
                "input_image_hash": analysis.image_sha256,
                "face_vector_hash": primary_face.vector_hash,
            },
            "discovered_match": match_record.to_dict(),
            "related_verified_content": [getattr(m, "candidate", m).post_url for m in related_verified_matches],
            "blockchain_proof": {
                "match_record_hash": receipt.match_record_hash,
                "network": receipt.network_name,
                "chain_id": receipt.chain_id,
                "contract_address": receipt.contract_address,
                "transaction_hash": receipt.tx_hash,
                "block_number": receipt.block_number,
                "explorer_url": receipt.explorer_url,
                "gas_used": receipt.gas_used,
                "notary_wallet": receipt.wallet_address,
            },
        }
        with open(save_receipt, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        console.print(f"[bold green][+] Match results saved to {save_receipt}[/bold green]")

    print_final_verification_section(
        is_social_winner=True,
        winner_or_top_web=top_result,
        identity_name=discovered_attribution,
        confidence_str=f"{identity_cand.confidence * 100:.0f}%" if identity_cand else None,
        receipt=receipt,
        verification_passed=outcome.is_valid,
    )

    console.print("\n[bold green]>>> Pipeline completed successfully! Tamper-evident record is immutably anchored on-chain.[/bold green]")


def verify_receipt_mode(receipt_path: str, network: str = DEFAULT_NETWORK, contract_address: str = ""):
    """Verifies a saved receipt against the on-chain registry without requiring input image."""
    print_banner()
    if not os.path.exists(receipt_path):
        console.print(f"[bold red]Error: Receipt file '{receipt_path}' not found.[/bold red]")
        sys.exit(1)

    print_step_header(5, f"Independent On-Chain Match Re-Verification ({receipt_path})")
    with console.status("[bold green]Loading receipt and querying smart contract on-chain state..."):
        verifier = ProofVerifier(network_key=network, contract_address=contract_address)
        outcome = verifier.verify_match_record(receipt_path, contract_address=contract_address)

    print_step5_match_verification(outcome)

    if outcome.is_valid:
        sys.exit(0)
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Face ID + Reverse Search + Blockchain Verification Pipeline"
    )
    parser.add_argument(
        "image_pos",
        nargs="?",
        default="",
        help="Optional positional path to input face image file",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default="",
        help="Path to input face image file",
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        default=DEFAULT_NETWORK,
        choices=["polygon_amoy", "sepolia", "local"],
        help="Target blockchain network",
    )
    parser.add_argument(
        "--contract",
        "-c",
        type=str,
        default="",
        help="Deployed FaceProofRegistry contract address",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold (default: 0.70)",
    )
    parser.add_argument(
        "--save-receipt",
        "-s",
        type=str,
        default="",
        help="Path to save output JSON receipt file (defaults to receipts/<image_name>_receipt.json)",
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default="",
        help="Optional direct social profile URL (offline dev testing only)",
    )
    parser.add_argument(
        "--live-demo",
        action="store_true",
        help="Strict evaluator live demo mode (requires deployed contract and live APIs)",
    )
    parser.add_argument(
        "--verify",
        "-v",
        type=str,
        default="",
        help="Path to receipt JSON to independently verify against the blockchain",
    )
    parser.add_argument(
        "--simulate-tamper",
        action="store_true",
        help="Simulates a local data tampering attack in Step 5 to demonstrate cryptographic tamper-detection",
    )
    parser.add_argument(
        "--no-blockchain",
        "--skip-blockchain",
        action="store_true",
        dest="skip_blockchain",
        help="Run pipeline through Step 3 only (skips on-chain anchoring and verification for quick testing)",
    )

    args = parser.parse_args()

    if args.verify:
        verify_receipt_mode(
            receipt_path=args.verify,
            network=args.network,
            contract_address=args.contract,
        )
        return

    image_path = args.image or args.image_pos
    if not image_path:
        console.print("[bold yellow]No input image specified via --image / -i.[/bold yellow]")
        try:
            image_path = input("Enter path to input face image (e.g. sample_images/who.jpg): ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Operation cancelled.[/bold red]")
            sys.exit(1)
        if not image_path:
            console.print("[bold red]Error: No image path provided. Please specify an image using --image <path>[/bold red]")
            sys.exit(1)

    run_pipeline(
        image_path=image_path,
        network=args.network,
        contract_address=args.contract,
        threshold=args.threshold,
        save_receipt=args.save_receipt,
        direct_social_url=args.url,
        live_demo=args.live_demo,
        simulate_tamper=args.simulate_tamper,
        skip_blockchain=args.skip_blockchain,
    )


if __name__ == "__main__":
    main()
