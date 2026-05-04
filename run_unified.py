import sys
import os
import time
import threading
import signal
from http.server import HTTPServer
from socketserver import ThreadingMixIn

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.getcwd())

from run_api import Handler, system, lattice
from src.daemon import LivingDatabaseDaemon

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def start_api():
    httpd = ThreadingHTTPServer(('', 8009), Handler)
    httpd.serve_forever()

def start_daemon(daemon_instance):
    daemon_instance.start()

def start_methodology_sim():
    while True:
        try:
            system.run_step()
            time.sleep(2)
        except Exception as e:
            print(f"[Sim Error] {e}")
            time.sleep(5)

if __name__ == "__main__":
    print("\n" + "█"*75)
    print("█  DATAG UNIFIED NEURAL STACK (API + DAEMON)  █")
    print("█  Status: SYNCHRONIZING MULTI-LAYER CONSCIOUSNESS  █")
    print("█"*75 + "\n")

    daemon_instance = LivingDatabaseDaemon()
    
    # Components as threads
    api_thread = threading.Thread(target=start_api, daemon=True)
    daemon_thread = threading.Thread(target=start_daemon, args=(daemon_instance,), daemon=True)
    sim_thread = threading.Thread(target=start_methodology_sim, daemon=True)

    print("[1/3] Starting API (Port 8009)...")
    api_thread.start()
    
    print("[2/3] Initiating Subconscious Pulse (11M Cell Daemon)...")
    daemon_thread.start()
    
    print("[3/3] Activating Methodology Evolution sim...")
    sim_thread.start()

    print("\n" + "="*75)
    print("  [SUCCESS] DataG is now fully ALIVE, looking, and thinking.")
    print("  Logs from all layers will be interleaved below.")
    print("  (Press Ctrl+C to safely hibernate the system)")
    print("="*75 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[Shutdown] Hibernation signal received.")
        print("[Shutdown] Persisting evolved cells to disk...")
        daemon_instance.running = False
        # Give daemon a moment to finish its last pulse and persist
        time.sleep(5)
        print("[Shutdown] Goodbye.")
        sys.exit(0)
