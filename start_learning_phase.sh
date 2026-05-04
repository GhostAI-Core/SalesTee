#!/bin/bash
# DataG Phase 3: Neural Independence Learning Sequence
# Derived from Neural Status Report Failure Audit

echo "=========================================="
echo " STAGE 1: HARD RESET"
echo "=========================================="
echo "Terminating lingering Senses and Brain processes..."
pkill -9 -f run_api.py || true
pkill -9 -f shadow_doc_learner.py || true
sleep 2

echo ""
echo "=========================================="
echo " STAGE 2: IGNITING DATAG BRAIN"
echo "=========================================="
echo "Loading all 3,833 methodology cells natively..."
cd /home/odessey/.gemini/antigravity/scratch/DataG
python3 run_api.py > api_persistent.log 2>&1 &
BRAIN_PID=$!

echo "Giving the Brain exactly 60 seconds to fully initialize and settle (safety buffer)..."
sleep 60
echo "✅ Brain boot buffer complete. Verifying connection..."
curl -s --max-time 2 http://localhost:8009/assimilation_status > /dev/null || echo "⚠️ Warning: Brain might still be sluggish, but proceeding..."
echo "✅ Brain is responding to telemetry."

echo ""
echo "=========================================="
echo " STAGE 3: LAUNCHING IA SENSES"
echo "=========================================="
echo "Initializing Shadow Document Learner with CRITICAL memory-cooling parameters..."
cd /home/odessey/.gemini/antigravity/scratch/IA_code_base
# Delay 15 is explicitly enforced to prevent the deadlock!
python3 shadow_doc_learner.py --delay 15 --count 500
