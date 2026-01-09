import os
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def check_data():
    """Check if data folder exists, download if not."""
    data_path = REPO_ROOT / "data"
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Data folder not found. Downloading from HuggingFace...")
        from scripts.download_data import download_data
        download_data()

def check_env():
    """Check if required environment variables are set."""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["GROQ_API_KEY", "COHERE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("Please create a .env file with the required variables.")
        print("Example:")
        print("  GROQ_API_KEY=your_groq_key")
        print("  COHERE_API_KEY=your_cohere_key")
        sys.exit(1)

def main():
    print("=" * 60)
    print("HUST RAG Assistant - Startup")
    print("=" * 60)
    
    # Check data
    check_data()
    
    # Check environment
    check_env()
    
    # Run Gradio app
    print("\nStarting Gradio server...")
    from core.gradio.user_gradio import demo, GRADIO_CFG
    
    demo.launch(
        server_name=GRADIO_CFG.server_host,
        server_port=GRADIO_CFG.server_port
    )

if __name__ == "__main__":
    main()
