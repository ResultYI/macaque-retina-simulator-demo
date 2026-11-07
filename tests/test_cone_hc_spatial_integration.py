from pathlib import Path
import unittest
import hashlib
import json
import numpy as np
from scipy import sparse
from retina_sim.assembly import RetinaBuilder
from retina_sim.cone_hc_spatial import select_spatial_A
from retina_sim.contracts import ContractError

ROOT=Path(__file__).resolve().parents[1]


class SpatialIntegration(unittest.TestCase):
    def test_mode_selection_preserves_H2_and_uniform(self):
        u=sparse.csr_matrix([[.5,.5,0],[.2,.3,.5]])
        b=sparse.csr_matrix([[.8,.2,0],[.4,.1,.5]])
        types=np.array(['H1','H2'])
        np.testing.assert_array_equal(select_spatial_A(u,b,types,'uniform').toarray(),u.toarray())
        selected=select_spatial_A(u,b,types,'biexponential').toarray()
        np.testing.assert_array_equal(selected[0],b.toarray()[0]);np.testing.assert_array_equal(selected[1],u.toarray()[1])
        with self.assertRaises(ContractError):select_spatial_A(u,b,types,'automatic')

    def test_config_build_keeps_local_B_and_unsupported_registry(self):
        models=[RetinaBuilder.build_research_cone_hc(ROOT,'configurations/research/cone_hc_'+mode+'.json')
                for mode in ('uniform','biexponential')]
        self.assertEqual(models[0].A.shape,(249,321));self.assertEqual((models[0].B-models[1].B).nnz,0)
        self.assertNotEqual(models[0].identity,models[1].identity)
        for model in models:
            self.assertLess(model.background_rhs,1e-8)
            np.testing.assert_array_equal(model.geometry['cone_parent_rows'][model.binding['unsupported_feedback']],[166128])
            self.assertEqual(model.state_layout['release_unit'],'dimensionless')
            self.assertFalse(model.config['production_qualified'])

    def test_frozen_default_provider_hash_is_unchanged(self):
        cfg=json.loads((ROOT/'configurations/hc_density_geometry_current.json').read_bytes())
        expected=cfg['bindings']['h1_h2_effective.py']['sha256']
        self.assertEqual(hashlib.sha256((ROOT/'retina_sim/h1_h2_effective.py').read_bytes()).hexdigest(),expected)


if __name__=='__main__':unittest.main()
