import subprocess
import sys

def main():
    """Run the Streamlit app"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Thanks for using PDF QA!")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()
