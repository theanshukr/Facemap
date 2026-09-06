"""
Rich Terminal UI Display Helper for Face ID + Blockchain Verification Pipeline.
Provides visual output, explainability breakdowns, search provenance,
candidate comparison tables, and tamper-evident verification certificates.
"""

import sys
from typing import List, Any, Optional, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Ensure UTF-8 output streams on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

console = Console(force_terminal=True, legacy_windows=False)


def print_banner():
    """Prints the pipeline title banner."""
    banner_text = (
        "[bold cyan]============================================================[/bold cyan]\n"
        "[bold cyan]          FACE ID + BLOCKCHAIN VERIFICATION PIPELINE        [/bold cyan]\n"
        "[bold cyan]============================================================[/bold cyan]\n"
        "[bold magenta]* Deep Neural Face ID + Live Reverse Search + Polygon Amoy Proof *[/bold magenta]"
    )
    console.print(Panel(banner_text, border_style="cyan", box=box.ROUNDED))


def print_step_header(step_num: int, step_title: str):
    """Prints a styled step header."""
    console.print(f"\n[bold blue]━━━ STEP {step_num}: {step_title.upper()} ━━━[/bold blue]")


def print_face_detection_summary(result):
    """Displays face detection and landmark statistics."""
    table = Table(title="[bold green]Deep Neural Face Detection & Embedding (YuNet + SFace)[/bold green]", box=box.ROUNDED)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Input File", str(result.image_path))
    table.add_row("Image SHA-256 (Provenance)", f"[yellow]{result.image_sha256}[/yellow]")
    table.add_row("Dimensions", f"{result.dimensions[0]} x {result.dimensions[1]} px")
    table.add_row("Faces Detected", f"[bold green]{len(result.faces)}[/bold green]")

    if result.primary_face:
        pf = result.primary_face
        table.add_row("Detection Model", f"[cyan]{pf.detector_model}[/cyan]")
        table.add_row("Embedding Model", f"[magenta]{pf.embedding_model}[/magenta]")
        table.add_row("Primary BBox", f"x={pf.bbox[0]}, y={pf.bbox[1]}, w={pf.bbox[2]}, h={pf.bbox[3]}")
        table.add_row("Confidence", f"{pf.confidence * 100:.1f}%")
        table.add_row("Vector Hash (128-d)", f"[magenta]{pf.vector_hash}[/magenta]")
        table.add_row("Landmarks Found", ", ".join(pf.landmarks.keys()))

    console.print(table)


def print_search_results(candidates: List[Any]):
    """Displays discovered social media and web candidates."""
    table = Table(title="[bold green]Reverse-Image Search Web & Social Discoveries[/bold green]", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3)
    table.add_column("Platform / Source", style="cyan", width=18)
    table.add_column("Category", style="yellow", width=14)
    table.add_column("Discovered Post / Page URL", style="magenta")
    table.add_column("Candidate Image", style="white", width=18)
    table.add_column("Engine", style="dim")

    for idx, c in enumerate(candidates, 1):
        img_badge = "[green]IMAGE AVAIL[/green]" if c.has_image else "[dim yellow]NO IMAGE[/dim yellow]"
        cat_badge = "[bold green]SOCIAL MEDIA[/bold green]" if c.is_social_media else "[dim]WEB DIRECTORY[/dim]"
        engine_str = getattr(c, "source_engine", "Reverse Image Search")
        table.add_row(
            str(idx),
            c.platform.upper(),
            cat_badge,
            c.post_url,
            img_badge,
            engine_str,
        )

    console.print(table)


def print_candidate_ranking_table(ranking_report: Any):
    """Displays all biometric candidates ranked by facial similarity score with social eligibility highlights."""
    ranked = ranking_report.ranked_candidates

    console.print(
        f"\n[bold white]Discovery Summary:[/bold white] "
        f"[green]Biometric Candidates: {len(ranked)}[/green] | "
        f"[yellow]Metadata-Only Discoveries: {ranking_report.metadata_only_count}[/yellow] | "
        f"[dim]Skipped Discoveries: {len(ranking_report.skipped_log)}[/dim]\n"
    )

    if not ranked:
        console.print("[bold yellow]No candidate images with detectable faces passed URL validation and biometric matching.[/bold yellow]")
        return

    table = Table(title="[bold green]REVERSE SEARCH CANDIDATE COMPARISON[/bold green]", box=box.ROUNDED)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Platform", style="cyan", width=12)
    table.add_column("Type", style="yellow", width=10)
    table.add_column("Similarity", style="bold yellow", width=11)
    table.add_column("Status / Selection", style="bold", width=22)
    table.add_column("Content URL", style="magenta")

    top_social_winner, _ = ranking_report.select_final_social_match()

    for idx, r in enumerate(ranked, 1):
        score_str = f"{r.similarity_score * 100:.2f}%"
        is_selected = (top_social_winner is not None and r.candidate.post_url == top_social_winner.candidate.post_url)

        if is_selected:
            status_str = "[bold green]✓ SELECTED WINNER[/bold green]"
        elif r.is_social_media:
            status_str = "[cyan]Social Candidate[/cyan]"
        else:
            status_str = "[dim yellow][EXCLUDED: NON-SOCIAL][/dim yellow]"

        table.add_row(
            str(idx),
            r.candidate.platform.upper(),
            r.content_type or "Post",
            score_str,
            status_str,
            r.candidate.post_url,
        )

    console.print(table)

    # Explainability on non-social exclusions
    has_higher_web = any(
        not r.is_social_media and top_social_winner is not None and r.similarity_score > top_social_winner.similarity_score
        for r in ranked
    )
    if has_higher_web:
        console.print("[dim yellow]ℹ Notice: Higher-similarity web results were excluded because the primary verification requirement mandates a genuine social-media post.[/dim yellow]\n")


def print_match_analysis(eval_result, threshold: float = 0.70):
    """Displays 'Why this match?' explainability decision breakdown."""
    sim = eval_result.similarity_score
    sim_pct = sim * 100
    th_pct = threshold * 100
    passed = sim >= threshold

    console.print("\n[bold cyan]━━━ MATCH EXPLAINABILITY ANALYSIS ━━━[/bold cyan]")
    console.print(f"Face detected:            [bold green]✓[/bold green]")
    console.print(f"Face embedding generated: [bold green]✓[/bold green] (OpenCV SFace 128-D)")
    console.print(f"Reverse-search candidate: [bold green]✓[/bold green] (Live Content Link)")
    console.print(f"Candidate face detected:  [bold green]✓[/bold green] ({eval_result.detected_faces_count} face(s) found)")
    console.print(f"\nFace Similarity:          [bold yellow]{sim_pct:.2f}%[/bold yellow] (Cosine metric)")
    console.print(f"Configured Threshold:     [white]{th_pct:.2f}%[/white]")
    console.print(f"\nDecision Engine:")
    if passed:
        console.print(f"[bold green]{sim_pct:.2f}% ≥ {th_pct:.2f}%[/bold green]")
        console.print("[bold green]→ FACE MATCH VERIFIED[/bold green]\n")
    else:
        console.print(f"[bold red]{sim_pct:.2f}% < {th_pct:.2f}%[/bold red]")
        console.print("[bold red]→ SIMILARITY BELOW THRESHOLD[/bold red]\n")


def print_match_evaluation(
    eval_result,
    social_url: str,
    is_social_media: bool = True,
    platform: str = "Social",
    content_type: str = "Post",
    url_status: str = "ACCESSIBLE",
    attribution: Optional[str] = None,
    selection_reason: Optional[str] = None,
):
    """Displays cross-verification match metrics for the selected top match."""
    table_title = "[bold green]━━━ VERIFIED SOCIAL MEDIA MATCH ━━━[/bold green]" if is_social_media else "[bold yellow]━━━ TOP MATCHED NON-SOCIAL DISCOVERY ━━━[/bold yellow]"
    table = Table(title=table_title, box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Verified Value", style="white")

    strat_label = "Highest Similarity Genuine Social Match (≥ 70%)" if is_social_media else "Top Matched Non-Social Discovery (≥ 70%)"
    table.add_row("Selection Strategy", f"[bold yellow]{strat_label}[/bold yellow]")
    if selection_reason:
        table.add_row("Selection Rationale", f"[bold green]{selection_reason}[/bold green]")

    name_display = attribution if (attribution and attribution.strip()) else "Not available"
    name_source = "Discovered Search Metadata" if (attribution and attribution.strip()) else "N/A"

    table.add_row("Name / Attribution", f"[bold yellow]{name_display}[/bold yellow]")
    table.add_row("Name Source", f"[dim]{name_source}[/dim]")
    table.add_row("Selected Platform", f"[bold cyan]{platform.upper()}[/bold cyan]")
    table.add_row("Selected Content Type", f"[yellow]{content_type}[/yellow]")
    table.add_row("Selected URL", f"[blue]{social_url}[/blue]")
    st_color = "green" if url_status == "ACCESSIBLE" else "yellow"
    table.add_row("URL Status", f"[{st_color}]{url_status}[/{st_color}]")
    table.add_row("Face Similarity", f"[bold green]{eval_result.similarity_score * 100:.2f}%[/bold green]")
    table.add_row("Euclidean Distance", f"{eval_result.euclidean_distance:.4f}")
    table.add_row("Candidate Faces", str(eval_result.detected_faces_count))
    badge = "[bold green]VERIFIED[/bold green]" if eval_result.is_match else "[bold red]FAILED[/bold red]"
    table.add_row("Face Verification", badge)

    console.print(table)


def print_search_provenance(
    engine: str,
    total_discovered: int,
    biometric_candidates: int,
    social_candidates: int,
    selected_platform: str,
    similarity_score: float,
    identity_candidate: Optional[Any] = None,
):
    """Displays search provenance and explainability."""
    table = Table(title="[bold green]SEARCH PROVENANCE & DISCOVERY PIPELINE[/bold green]", box=box.ROUNDED)
    table.add_column("Metric / Stage", style="cyan", width=26)
    table.add_column("Details", style="white")

    table.add_row("Search Engine", f"[bold yellow]{engine}[/bold yellow]")
    table.add_row("Candidates Discovered", f"[bold white]{total_discovered}[/bold white]")
    table.add_row("Biometric Candidates", f"[green]{biometric_candidates}[/green]")
    table.add_row("Social Candidates", f"[cyan]{social_candidates}[/cyan]")
    if identity_candidate and getattr(identity_candidate, "is_confident", False):
        src_domains = ", ".join(getattr(identity_candidate, "supporting_domains", [])[:3])
        table.add_row(
            "Identity Signal",
            f"[bold yellow]{identity_candidate.name}[/bold yellow] [dim]({identity_candidate.confidence * 100:.0f}% confidence across {identity_candidate.source_count} sources: {src_domains})[/dim]"
        )
    table.add_row("Selected Candidate", f"[bold yellow]{selected_platform.upper()}[/bold yellow]")
    table.add_row(
        "Selection Strategy",
        f"[bold green]Highest Similarity Match ({selected_platform.upper()}) → {similarity_score * 100:.2f}%[/bold green]"
    )

    console.print(table)


def print_related_matches_table(related_matches: List[Any]) -> None:
    """Displays secondary verified social posts belonging to the same person."""
    console.print("\n[bold cyan]━━━ RELATED DISCOVERED CONTENT (SAME PERSON) ━━━[/bold cyan]")
    if not related_matches:
        console.print("[dim yellow]No additional verified social posts or reels found meeting biometric criteria (similarity ≥ 70%).[/dim yellow]")
        return

    table = Table(title="[bold green]BIOMETRICALLY VERIFIED RELATED POSTS & REELS[/bold green]", box=box.ROUNDED)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Platform", style="cyan", width=12)
    table.add_column("Type", style="yellow", width=10)
    table.add_column("Similarity", style="bold green", width=12)
    table.add_column("Verification", style="bold green", width=14)
    table.add_column("Genuine Post / Reel URL", style="magenta")

    for idx, r in enumerate(related_matches, 1):
        plat = getattr(r, "platform", "")
        if not plat and hasattr(r, "candidate"):
            plat = getattr(r.candidate, "platform", "")
        plat_str = str(plat).upper()

        c_type = getattr(r, "content_type", "") or "Post"
        sim_score = getattr(r, "similarity_score", None)
        if sim_score is None and hasattr(r, "face_similarity"):
            sim_score = getattr(r, "face_similarity", 0.0)
        score_str = f"{float(sim_score) * 100:.2f}%" if sim_score is not None else "N/A"

        post_url = getattr(r, "post_url", "")
        if not post_url and hasattr(r, "candidate"):
            post_url = getattr(r.candidate, "post_url", "")

        table.add_row(
            str(idx),
            plat_str,
            c_type,
            score_str,
            "[bold green]VERIFIED[/bold green]",
            post_url,
        )

    console.print(table)


def print_step4_match_anchor(match_record, input_image_hash: str, receipt) -> None:
    """Displays Step 4 match record blockchain anchoring details with interactive explorer banner."""
    platform = getattr(match_record, "platform", "") or match_record.get("platform", "")
    content_type = getattr(match_record, "content_type", "") or match_record.get("content_type", "")
    post_url = getattr(match_record, "post_url", "") or match_record.get("post_url", "")
    cand_hash = getattr(match_record, "candidate_content_hash", "") or match_record.get("candidate_content_hash", "")
    match_hash = getattr(receipt, "match_record_hash", "") or receipt.payload_summary.get("match_record_hash", "")

    console.print("\n[bold cyan]Selected Match:[/bold cyan]")
    console.print(f"Platform: [bold yellow]{platform.upper()}[/bold yellow]")
    console.print(f"Content Type: [yellow]{content_type}[/yellow]")
    console.print(f"URL: [blue]{post_url}[/blue]")

    console.print("\n[bold cyan]Candidate Content Hash:[/bold cyan]")
    console.print(f"[dim white]{cand_hash}[/dim white]")

    console.print("\n[bold cyan]Match Record Hash (Primary Proof Key):[/bold cyan]")
    console.print(f"[bold magenta]{match_hash}[/bold magenta]")

    console.print("\n[bold cyan]Input Image Hash (Provenance Only):[/bold cyan]")
    console.print(f"[dim]{input_image_hash}[/dim]")

    console.print(f"\n[bold cyan]{receipt.network_name} Transaction:[/bold cyan]")
    console.print(f"[magenta]{receipt.tx_hash}[/magenta]")
    console.print(f"Block Number: [white]{receipt.block_number}[/white] | Gas Used: [dim]{receipt.gas_used:,}[/dim]")

    # Prominent Explorer Banner
    explorer_banner = (
        f"[bold green]🔗 [ View Transaction on PolygonScan ][/bold green]\n"
        f"[underline blue]{receipt.explorer_url}[/underline blue]"
    )
    console.print(Panel(explorer_banner, border_style="magenta", box=box.ROUNDED))

    console.print("\n[bold green]✅ VERIFIED MATCH ANCHORED ON-CHAIN[/bold green]\n")


def print_step5_match_verification(outcome) -> None:
    """Displays Step 5 independent match record on-chain re-verification."""
    console.print("\n" + "─" * 60)
    console.print("[bold cyan]STEP 5 — BLOCKCHAIN VERIFICATION[/bold cyan]\n")

    if outcome.is_valid:
        console.print(f"Network: [bold yellow]{outcome.network}[/bold yellow]\n")
        if outcome.tx_hash:
            console.print("Transaction:")
            console.print(f"[magenta]{outcome.tx_hash}[/magenta]\n")
        console.print("On-chain Match Record Hash:")
        console.print(f"[bold magenta]{outcome.on_chain_match_record_hash}[/bold magenta]\n")
        console.print("[bold white]Independent Verification:[/bold white]")
        console.print("[bold green]✓ On-chain proof found[/bold green]")
        console.print("[bold green]✓ Match record reconstructed[/bold green]")
        console.print("[bold green]✓ Local hash computed[/bold green]")
        console.print("[bold green]✓ Hash matches on-chain record[/bold green]")
        console.print("[bold green]✓ TAMPER-EVIDENT PROOF VERIFIED[/bold green]")
        console.print("─" * 60 + "\n")
        print_verification_certificate(outcome)
    else:
        console.print("[bold red]✗ MATCH RECORD VERIFICATION FAILED[/bold red]")
        console.print("[bold red]✗ LOCAL HASH DOES NOT MATCH ON-CHAIN HASH[/bold red]")
        console.print("[bold red]⚠️ POSSIBLE TAMPERING / DATA MODIFICATION DETECTED[/bold red]")
        if outcome.tamper_details:
            console.print(f"\n[dim yellow]Details: {outcome.tamper_details}[/dim yellow]")
        console.print("─" * 60 + "\n")


def print_verification_certificate(outcome) -> None:
    """Renders a Verification Certificate / Proof Summary box."""
    cand_short = outcome.local_candidate_content_hash[:16] + "..." if len(outcome.local_candidate_content_hash) > 16 else outcome.local_candidate_content_hash
    match_short = outcome.local_match_record_hash[:16] + "..." if len(outcome.local_match_record_hash) > 16 else outcome.local_match_record_hash

    table = Table(
        title="[bold yellow]╔══════════════════════════════════════════════════╗\n║           FACEMAP VERIFICATION CERTIFICATE       ║\n╚══════════════════════════════════════════════════╝[/bold yellow]",
        box=box.DOUBLE,
        border_style="green",
        show_header=False,
    )
    table.add_column("Property", style="bold cyan", width=24)
    table.add_column("Status / Value", style="bold white")

    table.add_row("Face Match", "[bold green]✓ VERIFIED[/bold green]")
    table.add_row("Similarity Score", f"[bold yellow]{outcome.similarity_score}[/bold yellow]")
    table.add_row("Social Post", "[bold green]✓ FOUND[/bold green]")
    table.add_row("Platform", f"[cyan]{outcome.platform.upper()}[/cyan]")
    table.add_row("Content Type", f"[white]{outcome.content_type}[/white]")
    table.add_row("Candidate Content Hash", f"[dim]{cand_short}[/dim]")
    table.add_row("Match Record Hash", f"[magenta]{match_short}[/magenta]")
    table.add_row("Blockchain", f"[yellow]{outcome.network}[/yellow]")
    table.add_row("Transaction", "[bold green]✓ CONFIRMED[/bold green]" if outcome.tx_hash else "[dim]ON-CHAIN[/dim]")
    table.add_row("On-Chain Verification", "[bold green]✓ PASSED[/bold green]")
    table.add_row("Security Guarantee", "[bold green]🔒 TAMPER-EVIDENT RECORD[/bold green]")

    console.print(table)


def print_blockchain_receipt(receipt):
    """Displays blockchain notarization receipt."""
    table = Table(title="[bold green]Polygon Amoy Smart Contract Notarization Receipt[/bold green]", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Blockchain Network", f"[bold yellow]{receipt.network_name} (Chain ID: {receipt.chain_id})[/bold yellow]")
    table.add_row("Match Record Hash", f"[bold magenta]{receipt.match_record_hash}[/bold magenta]")
    table.add_row("Candidate Content Hash", str(receipt.candidate_content_hash))
    table.add_row("Input Image Hash (Provenance)", f"[dim]{receipt.input_image_hash}[/dim]")
    table.add_row("Transaction Hash", f"[bold magenta]{receipt.tx_hash}[/bold magenta]")
    table.add_row("Block Number", str(receipt.block_number))
    table.add_row("Gas Used", f"{receipt.gas_used:,}")
    table.add_row("Notary Wallet", f"[dim]{receipt.wallet_address}[/dim]")
    table.add_row("Contract Address", str(receipt.contract_address))
    table.add_row("Block Explorer URL", f"[underline blue]{receipt.explorer_url}[/underline blue]")

    console.print(table)


def print_final_verification_section(
    is_social_winner: bool,
    winner_or_top_web: Any = None,
    identity_name: Optional[str] = None,
    confidence_str: Optional[str] = None,
    receipt: Any = None,
    verification_passed: bool = False,
    skip_blockchain: bool = False,
    skip_reason: str = "No verified social-media match",
):
    """Prints the unambiguous ## FINAL VERIFICATION summary block."""
    console.print("\n" + "─" * 60)
    console.print("[bold cyan]## FINAL VERIFICATION[/bold cyan]\n")

    if is_social_winner and winner_or_top_web:
        cand = getattr(winner_or_top_web, "candidate", winner_or_top_web)
        plat = str(getattr(cand, "platform", "Social")).upper()
        c_type = getattr(winner_or_top_web, "content_type", "") or getattr(cand, "content_type", "Post")
        sim_score = getattr(winner_or_top_web, "similarity_score", None)
        if sim_score is None and hasattr(winner_or_top_web, "evaluation"):
            sim_score = winner_or_top_web.evaluation.similarity_score
        if sim_score is None and hasattr(winner_or_top_web, "face_similarity"):
            sim_score = winner_or_top_web.face_similarity
        sim_val = float(sim_score) * 100 if sim_score is not None else 0.0
        post_url = getattr(cand, "post_url", "") or getattr(winner_or_top_web, "post_url", "")

        console.print("[bold green]Primary Social Match[/bold green]")
        console.print(f"Platform:        [bold yellow]{plat}[/bold yellow]")
        console.print(f"Content Type:    [white]{c_type}[/white]")
        console.print(f"Face Similarity: [bold green]{sim_val:.2f}%[/bold green]")
        console.print(f"URL:             [magenta]{post_url}[/magenta]\n")

        if identity_name and identity_name.strip() and identity_name.strip() != "Unresolved":
            conf = confidence_str if confidence_str else "High"
            console.print(f"Identity:        [bold yellow]{identity_name}[/bold yellow]")
            console.print(f"Confidence:      [cyan]{conf}[/cyan]\n")

        console.print("[bold white]Blockchain[/bold white]")
        if receipt:
            console.print(f"Network:            [yellow]{receipt.network_name}[/yellow]")
            console.print(f"Match Record Hash:  [magenta]{receipt.match_record_hash}[/magenta]")
            console.print(f"Transaction:        [magenta]{receipt.tx_hash}[/magenta]\n")
            console.print("[bold green]✓ Proof notarized[/bold green]")
            if verification_passed:
                console.print("[bold green]✓ On-chain verification passed[/bold green]")
            else:
                console.print("[bold red]❌ On-chain verification failed[/bold red]")
        elif skip_blockchain:
            console.print("[bold yellow]— Skipped[/bold yellow]")
            console.print(f"Reason:          [dim]{skip_reason}[/dim]")
    else:
        console.print("[bold yellow]No verified social-media match found.[/bold yellow]\n")

        if winner_or_top_web:
            cand = getattr(winner_or_top_web, "candidate", winner_or_top_web)
            source_name = str(getattr(cand, "platform", "Web")).upper()
            sim_score = getattr(winner_or_top_web, "similarity_score", None)
            if sim_score is None and hasattr(winner_or_top_web, "evaluation"):
                sim_score = winner_or_top_web.evaluation.similarity_score
            sim_val = float(sim_score) * 100 if sim_score is not None else 0.0
            post_url = getattr(cand, "post_url", "") or getattr(winner_or_top_web, "post_url", "")

            console.print("[bold white]Top Verified Web Discovery[/bold white]")
            console.print(f"Source:          [bold yellow]{source_name}[/bold yellow]")
            console.print(f"Face Similarity: [bold green]{sim_val:.2f}%[/bold green]")
            console.print(f"URL:             [magenta]{post_url}[/magenta]\n")

        console.print("[bold white]Blockchain[/bold white]")
        console.print("[bold yellow]— Skipped[/bold yellow]")
        console.print(f"Reason:          [dim]{skip_reason}[/dim]")

    console.print("─" * 60 + "\n")


# Clean section aliases for compact mode compatibility
def print_pipeline_header():
    print_banner()

def print_section_input_face(faces_count: int = 1, ready: bool = True):
    pass

def print_section_reverse_search(candidates_count: int):
    pass

def print_section_verification(social_matches_count: int, web_discoveries_count: int):
    pass

def print_section_identity(identity_name: Optional[str] = None, confidence_label: Optional[str] = None):
    pass

def print_section_social_result(winner: Any, related_matches: Optional[List[Any]] = None):
    pass

def print_section_web_fallback(top_web: Any):
    pass

def print_section_blockchain_success(receipt: Any):
    pass

def print_section_blockchain_skipped(reason: str):
    console.print(f"\n[bold yellow]⚡ Blockchain anchoring skipped ({reason}).[/bold yellow]")

def print_section_verification_outcome(outcome: Any):
    print_step5_match_verification(outcome)

def print_pipeline_footer():
    pass
