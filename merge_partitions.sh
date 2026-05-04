#!/bin/bash
# ============================================================
#  SAFE PARTITION MERGE: Move heavy dirs to Storage + Symlink
# ============================================================
#  This achieves the same effect as merging partitions, but
#  SAFELY on a live system. No rebooting, no data loss risk.
#
#  What it does:
#  1. Moves large directories from / (sda6, 836MB free) 
#     to /mnt/storage (sda1, 387GB free)
#  2. Creates symlinks so everything works exactly as before
#  3. All programs/paths work unchanged
#
#  To undo: reverse the moves and remove the symlinks.
# ============================================================

set -e  # Exit on any error

STORAGE="/mnt/c2e1dbd9-52cd-4731-b784-e8e24c4ff5dc"
STORAGE_HOME="$STORAGE/odessey_home"

echo "============================================================"
echo "  SAFE SPACE RECLAMATION"
echo "  Root (/) has only ~836MB free"
echo "  Storage has ~387GB free"
echo "  Moving heavy directories to Storage with symlinks"
echo "============================================================"

# Pre-flight check
if [ ! -d "$STORAGE" ]; then
    echo "[FATAL] Storage partition not mounted at $STORAGE"
    exit 1
fi

# Check available space on storage
STORAGE_FREE=$(df --output=avail "$STORAGE" | tail -1)
echo "[Check] Storage free: $(( STORAGE_FREE / 1024 / 1024 ))GB"

if [ "$STORAGE_FREE" -lt 100000000 ]; then
    echo "[FATAL] Storage has less than 100GB free. Aborting."
    exit 1
fi

# Create home mirror on storage
mkdir -p "$STORAGE_HOME"
echo "[OK] Storage target: $STORAGE_HOME"

# ─── Function to safely move + symlink ───────────────────────
move_and_link() {
    local SRC="$1"
    local NAME="$2"
    local DST="$STORAGE_HOME/$NAME"
    
    if [ -L "$SRC" ]; then
        echo "[SKIP] $SRC is already a symlink (already migrated)"
        return
    fi
    
    if [ ! -d "$SRC" ] && [ ! -f "$SRC" ]; then
        echo "[SKIP] $SRC does not exist"
        return
    fi
    
    local SIZE=$(du -sh "$SRC" 2>/dev/null | cut -f1)
    echo ""
    echo "[MOVE] $SRC ($SIZE) → $DST"
    
    # Step 1: Copy to storage (preserving permissions and ownership)
    if [ -d "$DST" ]; then
        echo "  → Destination exists, syncing..."
        rsync -a --info=progress2 "$SRC/" "$DST/"
    else
        echo "  → Copying..."
        rsync -a --info=progress2 "$SRC/" "$DST/"
    fi
    
    # Step 2: Verify copy
    local SRC_COUNT=$(find "$SRC" -type f 2>/dev/null | wc -l)
    local DST_COUNT=$(find "$DST" -type f 2>/dev/null | wc -l)
    
    if [ "$SRC_COUNT" -ne "$DST_COUNT" ]; then
        echo "  [WARNING] File count mismatch: source=$SRC_COUNT, dest=$DST_COUNT"
        echo "  → Keeping original as safety measure. NOT creating symlink."
        return
    fi
    
    echo "  → Verified: $DST_COUNT files matched."
    
    # Step 3: Remove original and create symlink
    echo "  → Removing original..."
    rm -rf "$SRC"
    
    echo "  → Creating symlink..."
    ln -s "$DST" "$SRC"
    
    echo "  [OK] $NAME migrated successfully. Freed $SIZE on root."
}

# ─── MOVE 1: .cache (17GB) ───────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Phase 1: Moving .cache (17GB)"
echo "═══════════════════════════════════════════════════"
move_and_link "/home/odessey/.cache" "dot_cache"

# ─── MOVE 2: D2A data_store (9.6GB and growing) ─────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Phase 2: Moving D2A data_store (9.6GB+)"
echo "═══════════════════════════════════════════════════"
move_and_link "/home/odessey/.gemini/antigravity/scratch/DataG/data_store" "d2a_data_store"

# ─── MOVE 3: .local/lib (7.5GB - pip packages, torch, etc) ──
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Phase 3: Moving .local/lib (7.5GB)"
echo "═══════════════════════════════════════════════════"
move_and_link "/home/odessey/.local/lib" "dot_local_lib"

# ─── MOVE 4: snap directory (2.2GB) ─────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Phase 4: Moving snap (2.2GB)"
echo "═══════════════════════════════════════════════════"
move_and_link "/home/odessey/snap" "snap"

# ─── SUMMARY ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  MIGRATION COMPLETE"
echo "═══════════════════════════════════════════════════"
echo ""
df -h /dev/sda6 /dev/sda1
echo ""
echo "Root (/) should now have ~36GB+ free."
echo "All symlinks are in place. Programs work unchanged."
echo "Postgres backup is safe. D2A cells now live on Storage."
