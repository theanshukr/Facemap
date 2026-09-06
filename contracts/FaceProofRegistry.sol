// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceProofRegistry
 * @dev Tamper-evident registry for verified face detection & social media reverse-image matches.
 * Anchors canonical match records on-chain with provenance metadata.
 */
contract FaceProofRegistry {
    
    struct ProofRecord {
        bytes32 matchRecordHash;       // SHA-256 digest of deterministic canonical match record (Primary Key)
        bytes32 inputImageHash;        // SHA-256 digest of original input image (Provenance only)
        bytes32 candidateContentHash;  // SHA-256 digest of discovered candidate media artifact
        string socialPostUrl;          // Discovered social media post URL
        uint256 similarityScore;       // Match similarity percentage scaled by 100 (e.g. 9450 = 94.50%)
        string metadataJson;           // JSON metadata payload (detection metrics, platform, engine version)
        uint256 timestamp;             // Block timestamp when registered
        address verifier;              // Notary wallet address
    }

    // Mapping from matchRecordHash to its ProofRecord
    mapping(bytes32 => ProofRecord) private _records;
    
    // Array of all registered match record hashes for enumeration
    bytes32[] private _allMatchRecordHashes;

    // Events (Maximum 3 indexed topics supported by EVM)
    event ProofRegistered(
        bytes32 indexed matchRecordHash,
        bytes32 indexed inputImageHash,
        bytes32 indexed candidateContentHash,
        string socialPostUrl,
        uint256 similarityScore,
        address verifier,
        uint256 timestamp
    );

    /**
     * @notice Registers a tamper-evident match proof on-chain.
     * @param matchRecordHash SHA-256 hash of the canonical match record
     * @param inputImageHash SHA-256 hash of the input image (provenance)
     * @param candidateContentHash SHA-256 hash of the discovered candidate image/media
     * @param socialPostUrl URL of the matching social media post
     * @param similarityScore Match confidence score scaled by 100
     * @param metadataJson Extended JSON metadata string
     */
    function registerProof(
        bytes32 matchRecordHash,
        bytes32 inputImageHash,
        bytes32 candidateContentHash,
        string calldata socialPostUrl,
        uint256 similarityScore,
        string calldata metadataJson
    ) external returns (bool) {
        require(matchRecordHash != bytes32(0), "Invalid match record hash");
        require(bytes(socialPostUrl).length > 0, "Social post URL cannot be empty");

        // If not already registered, track in match record hash list
        if (_records[matchRecordHash].timestamp == 0) {
            _allMatchRecordHashes.push(matchRecordHash);
        }

        _records[matchRecordHash] = ProofRecord({
            matchRecordHash: matchRecordHash,
            inputImageHash: inputImageHash,
            candidateContentHash: candidateContentHash,
            socialPostUrl: socialPostUrl,
            similarityScore: similarityScore,
            metadataJson: metadataJson,
            timestamp: block.timestamp,
            verifier: msg.sender
        });

        emit ProofRegistered(
            matchRecordHash,
            inputImageHash,
            candidateContentHash,
            socialPostUrl,
            similarityScore,
            msg.sender,
            block.timestamp
        );

        return true;
    }

    /**
     * @notice Verifies if a given matchRecordHash exists in the registry and returns proof details.
     * @param matchRecordHash SHA-256 hash of the canonical match record to look up
     */
    function getProof(bytes32 matchRecordHash) external view returns (
        bool exists,
        bytes32 inputImageHash,
        bytes32 candidateContentHash,
        string memory socialPostUrl,
        uint256 similarityScore,
        string memory metadataJson,
        uint256 timestamp,
        address verifier
    ) {
        ProofRecord memory record = _records[matchRecordHash];
        if (record.timestamp == 0) {
            return (false, bytes32(0), bytes32(0), "", 0, "", 0, address(0));
        }

        return (
            true,
            record.inputImageHash,
            record.candidateContentHash,
            record.socialPostUrl,
            record.similarityScore,
            record.metadataJson,
            record.timestamp,
            record.verifier
        );
    }

    /**
     * @notice Returns total number of registered proofs.
     */
    function getTotalProofs() external view returns (uint256) {
        return _allMatchRecordHashes.length;
    }

    /**
     * @notice Returns a match record hash by its registration index.
     */
    function getProofHashByIndex(uint256 index) external view returns (bytes32) {
        require(index < _allMatchRecordHashes.length, "Index out of bounds");
        return _allMatchRecordHashes[index];
    }
}
