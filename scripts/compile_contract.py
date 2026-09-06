import json
import solcx

def main():
    solcx.set_solc_version("0.8.20")
    compiled = solcx.compile_files(["contracts/FaceProofRegistry.sol"], output_values=["abi", "bin"])
    contract_key = "contracts/FaceProofRegistry.sol:FaceProofRegistry"
    abi = compiled[contract_key]["abi"]
    bin_bytecode = compiled[contract_key]["bin"]

    header = '"""\nABI definition and bytecode for FaceProofRegistry.\n"""\nimport json\n\n'
    abi_str = f"FACE_PROOF_REGISTRY_ABI = json.loads('''{json.dumps(abi)}''')\n\n"
    bin_str = f'FACE_PROOF_REGISTRY_BYTECODE = "0x{bin_bytecode}"\n'

    with open("src/blockchain/contract_abi.py", "w", encoding="utf-8") as f:
        f.write(header + abi_str + bin_str)

    print("Successfully compiled and updated src/blockchain/contract_abi.py!")

if __name__ == "__main__":
    main()
