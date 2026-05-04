"""
Living Database Daemon
=======================
The background process that keeps the entire database ALIVE.

It does three things continuously:
  1. PULSE — Rotate through cells in batches, letting them interact
  2. INGEST — Check incoming data for new concepts, create cells if unknown
  3. PERSIST — Save evolved cell states back to disk

This runs 24/7. The cells never stop evolving.
The Seagate drive is the "brain" — always connected, always alive.
"""

import os
import sys
import json
import time
import hashlib
import threading
import signal
import glob
import re
import numpy as np
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.living_cell import LivingCell

# ─── CONFIGURATION ────────────────────────────────────────────
SEAGATE_PATH = "/media/odessey/Seagate Expansion Drive/living_db"
CELL_DIR = os.path.join(SEAGATE_PATH, "cells")
INDEX_DIR = os.path.join(SEAGATE_PATH, "index")
STATE_FILE = os.path.join(SEAGATE_PATH, "daemon_state.json")

# Performance tuning
BATCH_SIZE = 200           # Cells to load per pulse cycle  
INTERACTION_PAIRS = 20     # Cell pairs to interact per pulse
PULSE_INTERVAL = 2.0       # Seconds between pulse cycles
PERSIST_INTERVAL = 30.0    # Seconds between disk writes
INGEST_INTERVAL = 5.0      # Seconds between ingestion checks

# ─── SHARD UTILITIES ─────────────────────────────────────────
def get_shard(cell_id: str) -> str:
    """Get 2-char hex shard directory for a cell."""
    return hashlib.md5(cell_id.encode()).hexdigest()[:2]

def cell_path(cell_id: str) -> str:
    """Full path for a cell file on the Seagate."""
    shard = get_shard(cell_id)
    return os.path.join(CELL_DIR, shard, f"{cell_id}.json")

def ensure_shard(cell_id: str):
    """Ensure the shard directory exists."""
    shard = get_shard(cell_id)
    os.makedirs(os.path.join(CELL_DIR, shard), exist_ok=True)


class LivingDatabaseDaemon:
    """
    The Living Database Daemon.
    Keeps the entire cell population alive, interacting, and evolving.
    """
    
    def __init__(self):
        self.running = False
        self.encoder = None      # Sentence transformer (loaded lazily)
        
        # In-memory working set (hot cells currently loaded)
        self.hot_cells = {}      # cell_id -> LivingCell
        self.dirty_cells = set() # cells that need persisting
        
        # Word/concept index — tracks what the database "knows"
        self.known_words = set()
        self.word_to_cells = defaultdict(set)  # word -> set of cell_ids
        
        # Ingestion queue
        self.ingest_queue = []
        self.ingest_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_cells_on_disk": 0,
            "pulses": 0,
            "interactions": 0,
            "cells_born": 0,
            "cells_evolved": 0,
            "uptime_start": time.time()
        }
        
        # Shard cursor — tracks which shard we're pulsing through
        self.shard_cursor = 0
        self.all_shards = [f"{i:02x}" for i in range(256)]
        
        print("=" * 60)
        print("  LIVING DATABASE DAEMON")
        print(f"  Storage: {SEAGATE_PATH}")
        print(f"  Pulse: every {PULSE_INTERVAL}s | Persist: every {PERSIST_INTERVAL}s")
        print("=" * 60)
    
    # ─── INITIALIZATION ──────────────────────────────────────
    
    def _load_encoder(self):
        """Load the sentence transformer for encoding new content."""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[Daemon] Loading neural encoder (all-MiniLM-L6-v2)...")
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                print("[Daemon] Encoder ready.")
            except Exception as e:
                print(f"[Daemon] Encoder failed: {e}")
    
    def _build_word_index(self):
        """Build the known-words index by scanning shard directories."""
        print("[Daemon] Building word index from cell IDs...")
        total = 0
        for shard in self.all_shards:
            shard_dir = os.path.join(CELL_DIR, shard)
            if not os.path.isdir(shard_dir):
                continue
            for fname in os.listdir(shard_dir):
                if fname.endswith(".json"):
                    cell_id = fname[:-5]
                    # Extract words from cell_id
                    words = re.findall(r'[a-zA-Z]{2,}', cell_id.lower())
                    for w in words:
                        self.known_words.add(w)
                        self.word_to_cells[w].add(cell_id)
                    total += 1
        
        self.stats["total_cells_on_disk"] = total
        print(f"[Daemon] Index built: {len(self.known_words):,} known words, {total:,} cells indexed.")
    
    def _count_cells(self) -> int:
        """Fast count of total cells on disk."""
        total = 0
        for shard in self.all_shards:
            shard_dir = os.path.join(CELL_DIR, shard)
            if os.path.isdir(shard_dir):
                total += len(os.listdir(shard_dir))
        return total
    
    # ─── CELL I/O ────────────────────────────────────────────
    
    def load_cell(self, cell_id: str) -> LivingCell:
        """Load a cell from disk into hot memory."""
        if cell_id in self.hot_cells:
            return self.hot_cells[cell_id]
        
        path = cell_path(cell_id)
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            cell = LivingCell.from_dict(data)
            self.hot_cells[cell_id] = cell
            return cell
        except Exception as e:
            return None
    
    def save_cell(self, cell: LivingCell):
        """Persist a cell to disk."""
        ensure_shard(cell.cell_id)
        path = cell_path(cell.cell_id)
        try:
            with open(path, 'w') as f:
                json.dump(cell.to_dict(), f)
        except Exception as e:
            print(f"[Daemon] Save error {cell.cell_id}: {e}")
    
    def create_cell(self, cell_id: str, content: str, source: str = "ingestion") -> LivingCell:
        """Birth a new living cell with encoded DNA."""
        cell = LivingCell(cell_id, content, source)
        
        # Encode content into DNA
        if self.encoder:
            cell.dna = self.encoder.encode(content, normalize_embeddings=True).astype(np.float32)
        
        self.hot_cells[cell_id] = cell
        self.dirty_cells.add(cell_id)
        self.stats["cells_born"] += 1
        
        # Update word index
        words = re.findall(r'[a-zA-Z]{2,}', content.lower())
        for w in words:
            self.known_words.add(w)
            self.word_to_cells[w].add(cell_id)
        
        return cell
    
    def evict_cold(self):
        """Evict least-recently-activated cells from hot memory to save RAM."""
        if len(self.hot_cells) < BATCH_SIZE * 3:
            return  # Not enough to worry about
        
        # Sort by last_activated, evict the oldest
        sorted_cells = sorted(self.hot_cells.values(), key=lambda c: c.last_activated)
        to_evict = sorted_cells[:BATCH_SIZE]
        
        for cell in to_evict:
            if cell.cell_id in self.dirty_cells:
                self.save_cell(cell)
                self.dirty_cells.discard(cell.cell_id)
            del self.hot_cells[cell.cell_id]
    
    # ─── PULSE (The Heartbeat) ───────────────────────────────
    
    def pulse(self):
        """
        One heartbeat of the living database.
        Loads a batch of cells, lets them interact, evolves them.
        """
        # Rotate through shards
        shard_hex = self.all_shards[self.shard_cursor % 256]
        self.shard_cursor += 1
        
        shard_dir = os.path.join(CELL_DIR, shard_hex)
        if not os.path.isdir(shard_dir):
            return
        
        # Load a batch from this shard
        files = os.listdir(shard_dir)
        if not files:
            return
        
        # Pick a random sample for this pulse
        sample_size = min(BATCH_SIZE, len(files))
        sample = np.random.choice(files, size=sample_size, replace=False)
        
        batch = []
        for fname in sample:
            if not fname.endswith('.json'):
                continue
            cell_id = fname[:-5]
            cell = self.load_cell(cell_id)
            if cell:
                batch.append(cell)
        
        if len(batch) < 2:
            return
        
        # --- INTERACTIONS ---
        # Random pairs interact
        pairs_done = 0
        for _ in range(min(INTERACTION_PAIRS, len(batch) // 2)):
            idx = np.random.choice(len(batch), size=2, replace=False)
            cell_a = batch[idx[0]]
            cell_b = batch[idx[1]]
            
            # Compute semantic similarity to determine interaction strength
            sim = np.dot(cell_a.dna, cell_b.dna)
            if sim > 0.1:  # Only interact if somewhat related
                cell_a.interact(cell_b, strength=float(sim))
                self.dirty_cells.add(cell_a.cell_id)
                self.dirty_cells.add(cell_b.cell_id)
                pairs_done += 1
        
        # --- DECAY ---
        for cell in batch:
            cell.decay(rate=0.01)
        
        self.stats["pulses"] += 1
        self.stats["interactions"] += pairs_done
        self.stats["cells_evolved"] += len(batch)
    
    # ─── INGEST (New Knowledge) ──────────────────────────────
    
    def ingest(self, text: str, source: str = "external"):
        """
        Process new information into the living database.
        Checks what the database already knows. Creates new cells for unknown concepts.
        Updates existing cells when they see familiar content.
        """
        if not text or not text.strip():
            return
        
        self._load_encoder()
        
        # Tokenize into meaningful chunks (words and phrases)
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        unique_words = set(words)
        
        new_words = unique_words - self.known_words
        known_hits = unique_words & self.known_words
        
        # Encode the full text
        if self.encoder:
            text_vector = self.encoder.encode(text, normalize_embeddings=True).astype(np.float32)
        else:
            text_vector = np.random.randn(LivingCell.DNA_DIM).astype(np.float32)
            text_vector /= np.linalg.norm(text_vector)
        
        # 1. Create cells for NEW words/concepts
        for word in new_words:
            cell_id = f"word_{word}"
            cell = self.create_cell(cell_id, f"concept: {word}", source)
            # Give it a DNA that's a blend of the word meaning and context
            if self.encoder:
                word_vec = self.encoder.encode(word, normalize_embeddings=True).astype(np.float32)
                cell.dna = word_vec * 0.7 + text_vector * 0.3
                norm = np.linalg.norm(cell.dna)
                if norm > 0:
                    cell.dna = cell.dna / norm
            self.dirty_cells.add(cell_id)
        
        # 2. Update existing cells that match (they "see" new context)
        for word in known_hits:
            cell_ids = list(self.word_to_cells.get(word, []))[:5]
            for cid in cell_ids:
                cell = self.load_cell(cid)
                if cell:
                    cell.learn(text_vector, learning_rate=0.0005)
                    cell.energy = min(100.0, cell.energy + 2.0)
                    self.dirty_cells.add(cid)
        
        # 3. Create a "memory" cell for this specific interaction
        mem_id = f"mem_{hashlib.md5(text[:100].encode()).hexdigest()[:12]}"
        mem_cell = self.create_cell(mem_id, text[:500], source)
        mem_cell.dna = text_vector.copy()
        self.dirty_cells.add(mem_id)
        
        if new_words:
            print(f"[Ingest] +{len(new_words)} new concepts | ~{len(known_hits)} updated | source: {source}")
        
        return {
            "new_concepts": len(new_words),
            "updated_cells": len(known_hits),
            "memory_cell": mem_id
        }
    
    # ─── PERSIST (Save to Disk) ──────────────────────────────
    
    def persist(self):
        """Save all dirty cells to disk."""
        if not self.dirty_cells:
            return 0
        
        saved = 0
        for cell_id in list(self.dirty_cells):
            cell = self.hot_cells.get(cell_id)
            if cell:
                self.save_cell(cell)
                saved += 1
        
        self.dirty_cells.clear()
        return saved
    
    # ─── QUERY (Ask the Living Database) ─────────────────────
    
    def query(self, text: str, top_k: int = 10) -> list:
        """
        Query the living database. 
        Finds relevant cells, activates them, returns their combined response.
        """
        self._load_encoder()
        
        if self.encoder:
            query_vec = self.encoder.encode(text, normalize_embeddings=True).astype(np.float32)
        else:
            return []
        
        # Find candidate cells from word index
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        candidate_ids = set()
        for w in words:
            candidate_ids.update(self.word_to_cells.get(w, set()))
        
        # Score candidates by DNA similarity
        scored = []
        for cid in list(candidate_ids)[:500]:  # Cap for performance
            cell = self.load_cell(cid)
            if cell is not None and np.linalg.norm(cell.dna) > 0:
                sim = float(np.dot(query_vec, cell.dna))
                if sim > 0.15:
                    # Activate the cell — it processes the query
                    output = cell.activate(query_vec)
                    scored.append({
                        "cell_id": cid,
                        "content": cell.content[:200],
                        "similarity": sim,
                        "energy": cell.energy,
                        "activations": cell.activation_count,
                        "connections": len(cell.connections)
                    })
                    self.dirty_cells.add(cid)
        
        # Sort by similarity
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
    
    # ─── MAIN LOOP ───────────────────────────────────────────
    
    def start(self):
        """Start the daemon. Runs until killed."""
        self.running = True
        
        # Setup directories
        os.makedirs(CELL_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)
        for shard in self.all_shards:
            os.makedirs(os.path.join(CELL_DIR, shard), exist_ok=True)
        
        # Load encoder
        self._load_encoder()
        
        # Build word index
        self._build_word_index()
        
        # Signal handler for graceful shutdown
        def shutdown(signum, frame):
            print("\n[Daemon] Shutdown signal received. Persisting state...")
            self.running = False
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        
        print(f"\n[Daemon] ALIVE — {self.stats['total_cells_on_disk']:,} cells on disk")
        print(f"[Daemon] Known vocabulary: {len(self.known_words):,} words")
        print(f"[Daemon] Starting pulse loop...\n")
        
        last_persist = time.time()
        last_status = time.time()
        
        while self.running:
            try:
                # 1. PULSE — let cells interact
                self.pulse()
                
                # 2. PROCESS INGEST QUEUE
                with self.ingest_lock:
                    queue_copy = self.ingest_queue[:]
                    self.ingest_queue.clear()
                
                for item in queue_copy:
                    self.ingest(item["text"], item.get("source", "queue"))
                
                # 3. PERSIST dirty cells periodically
                now = time.time()
                if now - last_persist > PERSIST_INTERVAL:
                    saved = self.persist()
                    if saved > 0:
                        print(f"[Persist] Saved {saved} evolved cells to Seagate")
                    last_persist = now
                    
                    # Evict cold cells from RAM
                    self.evict_cold()
                
                # 4. STATUS report every 60s
                if now - last_status > 60:
                    uptime = now - self.stats["uptime_start"]
                    print(f"[Status] Uptime: {uptime/3600:.1f}h | "
                          f"Pulses: {self.stats['pulses']} | "
                          f"Interactions: {self.stats['interactions']} | "
                          f"Born: {self.stats['cells_born']} | "
                          f"Hot: {len(self.hot_cells)} | "
                          f"Dirty: {len(self.dirty_cells)}")
                    last_status = now
                
                time.sleep(PULSE_INTERVAL)
                
            except Exception as e:
                print(f"[Daemon] Pulse error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
        
        # Final persist on shutdown
        print("[Daemon] Final persist...")
        saved = self.persist()
        print(f"[Daemon] Saved {saved} cells. Goodbye.")
    
    def queue_ingest(self, text: str, source: str = "api"):
        """Thread-safe: add text to the ingestion queue."""
        with self.ingest_lock:
            self.ingest_queue.append({"text": text, "source": source})


# ─── ENTRY POINT ─────────────────────────────────────────────

if __name__ == "__main__":
    daemon = LivingDatabaseDaemon()
    daemon.start()
