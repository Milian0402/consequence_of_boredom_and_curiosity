"""Use the established worker with independent predeclared confirmation seeds."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study
study.SEEDS['rank48-arm-confirm']=(73918261,498113807)
study.TILE_PHASES+=('rank48-arm-confirm',)
study.main()
