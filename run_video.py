"""
run_video.py
────────────
General video runner entrypoint. Automatically delegates to run_full_video.py.

Usage:
    python run_video.py
    python run_video.py --video videos/Wienerschnitzel_Sacramento_CA_95818__camA__2026_07_04_12_to_13_PDT.mkv
    python run_video.py --video videos/wienerschnitzel_10m.mkv
"""

from run_full_video import main

if __name__ == "__main__":
    main()
