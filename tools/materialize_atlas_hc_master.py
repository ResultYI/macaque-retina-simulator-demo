"""Build the limited approved joint HC geometry once; no dynamics or fallback."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retina_sim.source_atlas import SourceDensityAtlas
from retina_sim.atlas_hc import AtlasHCProvider
from retina_sim.atlas_hc_master import materialize_hc_master

if __name__=='__main__':
    atlas=SourceDensityAtlas(ROOT)
    print(json.dumps(materialize_hc_master(ROOT,atlas.config,AtlasHCProvider(ROOT).density),ensure_ascii=False))
