"""
Migrate cells from internal drive → Seagate Living Database
============================================================
Converts static JSON cells into living cells with real LoRA adapter weights.
Reads from both flat and sharded storage.
Writes to Seagate in sharded format.
"""

import os
import sys
import json
import glob
import time
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.living_cell import LivingCell

# Source: internal drive data_store
SOURCE_DIR = "/home/odessey/.gemini/antigravity/scratch/DataG/data_store/cells"

# Destination: Seagate
DEST_DIR = "/media/odessey/Seagate Expansion Drive/living_db/cells"

def get_shard(cell_id: str) -> str:
    return hashlib.md5(cell_id.encode()).hexdigest()[:2]

def dest_path(cell_id: str) -> str:
    shard = get_shard(cell_id)
    return os.path.join(DEST_DIR, shard, f"{cell_id}.json")

def main():
    print("=" * 60)
    print("  MIGRATE TO SEAGATE: Static Cells → Living Cells")
    print("=" * 60)
    
    # Load encoder for DNA generation
    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        print("[Encoder] Loading all-MiniLM-L6-v2...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("[Encoder] Ready.")
    except Exception as e:
        print(f"[Encoder] Not available: {e}")
    
    # Setup destination shards
    for i in range(256):
        os.makedirs(os.path.join(DEST_DIR, f"{i:02x}"), exist_ok=True)
    
    # Collect source files
    flat_files = glob.glob(os.path.join(SOURCE_DIR, "*.json"))
    shard_files = glob.glob(os.path.join(SOURCE_DIR, "*", "*.json"))
    all_files = flat_files + shard_files
    
    total = len(all_files)
    print(f"\n[Source] {total:,} cell files found.")
    print(f"[Dest] {DEST_DIR}")
    
    migrated = 0
    skipped = 0
    errors = 0
    batch_contents = []
    batch_cells = []
    ENCODE_BATCH = 256
    
    t0 = time.time()
    
    for filepath in all_files:
        try:
            # Check if already migrated
            with open(filepath, 'r') as f:
                old_data = json.load(f)
            
            cell_id = old_data.get("id", os.path.basename(filepath)[:-5])
            dp = dest_path(cell_id)
            
            if os.path.exists(dp):
                skipped += 1
                if (migrated + skipped) % 500000 == 0:
                    print(f"  → Progress: {migrated + skipped:,} / {total:,} (new: {migrated:,}, skip: {skipped:,})")
                continue
            
            # Extract content
            content = old_data.get("content", "")
            if not content:
                raw = old_data.get("raw_data", {})
                content = " | ".join(f"{k}: {v}" for k, v in raw.items() if v) if raw else cell_id
            
            source_table = old_data.get("source_table", "cortex")
            
            # Create living cell
            cell = LivingCell(cell_id, content[:500], source_table)
            
            # Reuse existing DNA if available
            old_state = old_data.get("state", {})
            if old_state.get("semantic_dna"):
                cell.dna = np.array(old_state["semantic_dna"], dtype=np.float32)
            else:
                # Queue for batch encoding
                batch_contents.append(content[:200])
                batch_cells.append(cell)
            
            # Save to Seagate
            with open(dp, 'w') as f:
                json.dump(cell.to_dict(), f)
            
            migrated += 1
            
            # Batch encode accumulated cells
            if len(batch_contents) >= ENCODE_BATCH and encoder:
                vectors = encoder.encode(batch_contents, normalize_embeddings=True, show_progress_bar=False)
                for bc, bv in zip(batch_cells, vectors):
                    bc.dna = bv.astype(np.float32)
                    # Re-save with DNA
                    with open(dest_path(bc.cell_id), 'w') as f:
                        json.dump(bc.to_dict(), f)
                batch_contents.clear()
                batch_cells.clear()
            
            if (migrated + skipped) % 100000 == 0:
                elapsed = time.time() - t0
                rate = (migrated + skipped) / elapsed if elapsed > 0 else 0
                eta = (total - migrated - skipped) / rate / 60 if rate > 0 else 0
                print(f"  → Progress: {migrated + skipped:,} / {total:,} | "
                      f"new: {migrated:,} | skip: {skipped:,} | "
                      f"rate: {rate:.0f}/s | ETA: {eta:.0f} min")
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [Error] {filepath}: {e}")
    
    # Final batch encode
    if batch_contents and encoder:
        vectors = encoder.encode(batch_contents, normalize_embeddings=True, show_progress_bar=False)
        for bc, bv in zip(batch_cells, vectors):
            bc.dna = bv.astype(np.float32)
            with open(dest_path(bc.cell_id), 'w') as f:
                json.dump(bc.to_dict(), f)
    
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"  MIGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Migrated:  {migrated:,} living cells")
    print(f"  Skipped:   {skipped:,} (already on Seagate)")
    print(f"  Errors:    {errors}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Location:  {DEST_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
