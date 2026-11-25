"""Use the common numerical, thread and timing checks with independent screen seeds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import study

study.SEEDS["finite-ranks"] = (35485721, 93392083)
study.TILE_PHASES += ("finite-ranks",)
study.main()
