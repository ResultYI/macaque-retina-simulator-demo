"""Focused entry/source/overwrite checks; no model, optical forward or fitting."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from retina_sim.contracts import ContractError
from tools import prepare_position_s0_v1 as entry


def fresh_output():
    return entry.ROOT/'reports'/('position_entry_test_'+uuid4().hex)


class PositionS0EntryTests(TestCase):
    def test_default_cli_is_read_only_and_reuses_near_geometry(self):
        output=fresh_output()
        center=entry.read(entry.NEAR/'GEOMETRY_SELECTION.json')['center_deg']
        args=['--center-deg',*map(str,center),'--output',str(output)]
        self.assertIsNone(entry.parser().parse_args(args).execute_step)
        with patch.object(entry.subprocess,'Popen',side_effect=AssertionError('Dry-run started a process')):
            with redirect_stdout(StringIO()):
                plan=entry.main(args)
        old=entry.read(entry.NEAR/'GEOMETRY_SELECTION.json')
        self.assertEqual(plan['target_parent_group_row'],old['selected_parent_group_row'])
        self.assertEqual(plan['target_type'],old['selected_type'])
        self.assertEqual(plan['actual_eccentricity_deg'],old['actual_eccentricity_deg'])
        self.assertFalse(output.exists())
        self.assertAlmostEqual(plan['parameter_provenance']['rod']['background_rstar_s'],.7450980392156863)

    def test_far_center_changes_only_position_identity(self):
        near=entry.preflight([-.04510050976293674,-.045844909154906355],fresh_output())
        far=entry.preflight([4.73,0.],fresh_output())
        self.assertEqual(far['actual_eccentricity_deg'],4.73)
        self.assertNotEqual(near['experiment_id'],far['experiment_id'])
        self.assertEqual(near['parameter_provenance'],far['parameter_provenance'])
        self.assertGreater(sum(far['core_RGC_centroid_counts'].values()),0)

    def test_dispatch_reuses_existing_algorithms_and_limits(self):
        plan=entry.preflight([4.73,0.],fresh_output());commands=entry.commands(plan)
        self.assertIn('--parent-construction',commands['construct'])
        self.assertIn(str(entry.PARENT),commands['construct'])
        self.assertIn('--selection-template',commands['construct'])
        self.assertIn('--scene-anchor',commands['optical_setup'])
        for command in commands.values():
            self.assertEqual(command[command.index('--max-rss-gib')+1],'24')
            self.assertFalse(any('run_cross_position_rgc' in arg for arg in command))
        with self.assertRaisesRegex(ContractError,'preceding explicit step'):
            entry.execute(plan,'initialize')
        self.assertFalse(Path(plan['output']).exists())

    def test_historical_output_and_invalid_centers_are_rejected(self):
        with self.assertRaisesRegex(ContractError,'historical results'):
            entry.preflight([4.73,0.],entry.ROOT/'reports/cone_aperture_eccentricity/HC_new_v1')
        with self.assertRaisesRegex(ContractError,'finite'):
            entry.preflight([np.nan,0.],fresh_output())
        with self.assertRaisesRegex(ContractError,'no existing RGC'):
            entry.preflight([100.,100.],fresh_output())

    def test_changed_S0_and_unchanged_native_regression(self):
        with TemporaryDirectory(prefix='position_S0_binding_') as directory:
            altered=Path(directory)/'anchor.json';altered.write_text('{}',encoding='utf-8')
            with patch.object(entry,'SCENE',altered):
                with self.assertRaisesRegex(ContractError,'S0 identity changed'):
                    entry.preflight([4.73,0.],fresh_output())
        native=entry.ROOT/'reports/isetbio_native_aperture_convergence_20261029_v1'
        identity=entry.read(native/'IMPLEMENTATION_IDENTITY.json');result=entry.read(native/'VALIDATION.json')
        self.assertEqual(result['status'],'PASS')
        for name,expected in identity['sources_sha256'].items():
            if name.endswith('.m'):
                self.assertEqual(entry.digest(entry.ROOT/name),expected)
        self.assertEqual(result['uniform']['maximum_capture_relative_error'],0)
        self.assertEqual(result['spatial']['maximum_capture_relative_error'],0)


if __name__=='__main__':
    main()
