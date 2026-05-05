import subprocess
import time
import sys
import shutil
from pathlib import Path

# Setup relative paths dynamically
BASE_DIR = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = BASE_DIR / 'federated' / 'fl_server.py'
CLIENT_SCRIPT = BASE_DIR / 'federated' / 'fl_client.py'

def launch_federation():
    print(">>> Starting Federated Learning Orchestration Pipeline\n")
    
    # Clean up stale keys from previous runs
    keys_dir = BASE_DIR / 'keys'
    if keys_dir.exists():
        shutil.rmtree(keys_dir)
        
    processes = []
    
    try:
        # 1. Start the FL Server
        print(">>> [1/4] Launching FL Server...")
        server_process = subprocess.Popen([sys.executable, str(SERVER_SCRIPT)])
        processes.append(('Server', server_process))
        
        # Wait 3 seconds to ensure server is securely bound to port 8080
        time.sleep(3)
        
        # 2. Start FL Clients
        for i, node_id in enumerate([1, 2, 3], start=2):
            print(f">>> [{i}/4] Launching FL Client Node {node_id}...")
            client_process = subprocess.Popen([
                sys.executable, 
                str(CLIENT_SCRIPT), 
                "--node_id", str(node_id)
            ])
            processes.append((f'Client {node_id}', client_process))
            
            # Wait 3 seconds between each client launch
            time.sleep(3)
            
        print("\n>>> All processes successfully launched.")
        print(">>> Waiting for federated training to finish... (This may take several minutes)\n")
        
        # 3. Wait for all processes to complete
        # Flower server automatically terminates after reaching the specified num_rounds
        # which subsequently triggers clean exit on connected clients.
        for name, p in processes:
            p.wait()
            
        print("\n" + "="*50)
        print("Federation complete")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n[WARNING] Keyboard interrupt detected. Force closing all FL processes...")
        for name, p in processes:
            p.terminate()
        print("All processes terminated.")

if __name__ == "__main__":
    launch_federation()
