import sys
from pathlib import Path

# Add project root (parent directory) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app

app = create_app()

if __name__ == "__main__":
    print("==================================================")
    print("Multi-Modal Fetal Risk Assessment System Starting")
    print("Dashboard URL: http://127.0.0.1:5000")
    print("Academic Research Prototype (Not for Clinical Use)")
    print("==================================================")
    app.run(host="0.0.0.0", port=5000, debug=True)
