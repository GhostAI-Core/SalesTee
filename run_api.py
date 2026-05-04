import sys
import os
import time
import json
import warnings
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading

# Suppress remaining deprecation warnings from dependencies
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Use __file__-relative path so imports work from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from system import AgenticSystem
from neural.liquid_adapter import LiquidAdapter
from substrate.nca_lattice import NCALattice
from neural.lora import LoRAAdapter

system = AgenticSystem(hdc_dim=384)
lattice = NCALattice(rows=10, cols=10)

# 🧠 Global Activity Log for UI Streaming
ACTIVITY_LOG = []
LATEST_COMPARISON = {
    "teacher": {},
    "student": {},
    "similarity": 0.0,
    "filename": "Awaiting Signal...",
    "timestamp": ""
}

def log_activity(msg):
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    ACTIVITY_LOG.insert(0, formatted)
    if len(ACTIVITY_LOG) > 50:
        ACTIVITY_LOG.pop()

tools = {
    "efficiency_expert": "Optimizes energy.",
    "repair_nanite": "Anomalous structural repair.",
    "explorer": "Semantic data discovery."
}
for label, desc in tools.items():
    adapter = LiquidAdapter(label, rank=16, dim_in=512, dim_out=512)
    system.orchestrator.register_adapter(adapter)
    system.register_tool(label, desc, LiquidAdapter)

def run_sim():
    while True:
        system.run_step()
        time.sleep(2)

threading.Thread(target=run_sim, daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/telemetry':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            telemetry = system.get_telemetry()
            stats = lattice.get_structure_stats()
            
            alerts = []
            anomaly_count = 0
            for cid, state in telemetry.items():
                if state.get("status") == "critical_failure":
                    anomaly_count += 3
                    alerts.insert(0, f"⚡ CRITICAL ALARM: Cell {cid} failed to organically repair.")
                    continue
                if state.get("energy", 100) < 50:
                    alerts.append(f"Cell {cid} reporting low energy ({state.get('energy')}%)")

            resp = {
                "entropy": stats["system_entropy"],
                "alerts": alerts,
                "anomaly_count": anomaly_count,
                "system_energy": sum(c.get("energy", 100) for c in telemetry.values()) / max(len(telemetry), 1),
                "total_records": len(telemetry),
                "active_adapters": len(system.orchestrator.gpu_cache),
                "total_adapters": len(system.orchestrator.adapter_registry),
                "activity_log": ACTIVITY_LOG
            }
            self.wfile.write(json.dumps(resp).encode())
            return

        # 🟢 DASHBOARD SERVING (FIX)
        if self.path in ['/', '/index.html', '/dashboard.html']:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open('dashboard.html', 'rb') as f:
                self.wfile.write(f.read())
            return

        if self.path == '/assimilation_status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            status = system.get_assimilation_status()
            self.wfile.write(json.dumps(status).encode())
            return
        
        if self.path == '/comparison':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(LATEST_COMPARISON).encode())
            return
        
        # Catch-all
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"NOT_FOUND")

    def do_POST(self):
        if self.path == '/assimilate':
            # Parse content-type without deprecated cgi module
            msg = Message()
            msg['Content-Type'] = self.headers.get('content-type', 'application/json')
            ctype = msg.get_content_type()
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length))
            try:
                log_activity(f"Assimilating NEW record... DNA Encoding active.")
                result = system.assimilate(message)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                cell_id = result[0].cell_id if isinstance(result, list) else result.cell_id
                log_activity(f"SUCCESS: Grafted {cell_id} into neural ecosystem.")
                self.wfile.write(json.dumps({"status": "success", "cell_id": cell_id}).encode())
            except Exception as e:
                log_activity(f"ERROR: Assimilation failed: {str(e)}")
                self.send_response(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == '/access':
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length))
            goal = message.get("goal", "")
            try:
                log_activity(f"Intent Engine: Accessing '{goal[:30]}...'")
                results = system.access(goal)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(results).encode())
            except Exception as e:
                self.send_response(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == '/encode':
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length))
            text = message.get("text", "")
            try:
                vec = system.hdc.encode(text)
                import numpy as np
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8: vec = vec / norm
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"vector": vec.tolist()}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == '/predict':
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length))
            query = message.get("query", "")
            try:
                log_activity(f"SHADOW: Predicting '{query[:30]}...'")
                result = system.predict(query)
                n_cells = len(result.get('activated_cells', []))
                if result.get('prediction') is not None:
                    log_activity(f"NEURAL: Activated {n_cells} methodology cells for prediction.")
                else: log_activity(f"NEURAL: No cells above threshold. Best similarity too low.")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == '/comparison':
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length))
            global LATEST_COMPARISON
            LATEST_COMPARISON = {
                "teacher": message.get("teacher", {}),
                "student": message.get("student", {}),
                "ia_truth": message.get("ia_truth", {}),
                "similarity": message.get("similarity", 0.0),
                "filename": message.get("filename", "unknown"),
                "timestamp": time.strftime("%H:%M:%S")
            }
            log_activity(f"SHADOW FEED UPDATED: {LATEST_COMPARISON.get('filename')}")
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "received"}).encode())
            
        elif self.path == '/learn_signal':
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length))
            cell_ids = message.get("cell_ids", [])
            similarity = message.get("similarity", 0.0)
            try:
                log_activity(f"LEARNING: Adjusting {len(cell_ids)} cell weights | Sim: {similarity:.1%}")
                result = system.learn_from_signal(cell_ids, message.get("target_vector", []), similarity)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        self.end_headers()

    def log_message(self, format, *args): pass

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer): daemon_threads = True

if __name__ == "__main__":
    print("MONOLITHIC SERVER STARTING ON PORT 8009...")
    ThreadingHTTPServer(('0.0.0.0', 8009), Handler).serve_forever()
