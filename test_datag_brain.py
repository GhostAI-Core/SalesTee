import requests
import json
import os
import random
#import PyPDF2

# Connect strictly to the already-running DataG port
DATAG_URL = "http://localhost:8009"
CV_DIR = "/home/odessey/.gemini/antigravity/scratch/DataG/cv_repository"

def extract_text(filepath):
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except:
        return ""

def test_independence():
    files = [f for f in os.listdir(CV_DIR) if f.endswith('.pdf')]
    if not files:
        print("No CVs found in the repository.")
        return
        
    cv = random.choice(files)
    print("=" * 60)
    print(f"🧠 INITIATING DATAG PHASE 3 (INDEPENDENCE TEST)")
    print(f"📄 Target: {cv}")
    print("=" * 60)
    print("\nExtracting raw text from CV...")
    text = extract_text(os.path.join(CV_DIR, cv))
    
    # By sending this to /predict, we bypass the 70B Teacher and the 8B Senses completely.
    # We are asking the DataG biological memory to parse the document using ONLY 
    # the Methodology Cells it has successfully assimilated so far.
    # There are NO token limits here because it uses High-Dimensional Vector mathematics.
    print(f"📡 Querying DataG Neural Engine strictly on Port 8009...\n")
    
    try:
        res = requests.post(f"{DATAG_URL}/predict", json={"query": f"parse document: {text}"}, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            if data.get("prediction"):
                print("🟩 INDEPENDENT PREDICTION COMPLETE 🟩\n")
                print(data.get("prediction_text", "No readable output trace."))
                
                cells = data.get("activated_cells", [])
                print(f"\n🧬 DataG achieved this by activating {len(cells)} stored Methodology Cells:")
                for c in cells:
                    print(f"   - {c}")
            else:
                 print("⚠️ DATAG REJECTED PREDICTION: " + str(data.get("error", "Unknown error")))
                 print("Reason: DataG's biological memory doesn't have enough methodology cells yet. Let the Shadow feed run longer!")
        else:
            print(f"Error {res.status_code}: {res.text}")
    except requests.exceptions.ConnectionError:
        print("CRITICAL ERROR: DataG is not currently running on port 8009.")

if __name__ == "__main__":
    test_independence()
