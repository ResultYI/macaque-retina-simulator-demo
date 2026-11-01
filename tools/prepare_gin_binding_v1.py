"""Prepare a real GIN target and decoded source input; never start neural work."""
from pathlib import Path
import argparse,hashlib,json,time,unittest
import cv2
import numpy as np
from retina_sim.natural_movie_supervision import file_sha,read_author_fixed_rf_parameters,read_raw_spike_tables,source_roster,bin_live_trial,frame_index_for_ticks
from retina_sim.gin_source_binding_v1 import source_display_lms,admission_blockers,require_neural_binding

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/source_packages/WP1_20261031/Schottdorf_Lee_xage77'
PREPARED=ROOT/'reports/RGC_DATASET_TARGET_MATRIX_V1/source_preparation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(out: Path) -> None:
    began=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    index=read(PREPARED/'GIN_SOURCE_INDEX.json');roster=source_roster(SOURCE/'README.md')
    if file_sha(SOURCE/'README.md')!=index['readme_sha256'] or file_sha(PREPARED/'GIN_RAW_SPIKES.npz')!=index['arrays_sha256']:
        raise ValueError('Existing prepared source identity changed')
    cell='69#6';records=[x for x in index['records'] if x['source_cell_key']==cell and not x['duplicate']]
    if len(records)!=2:raise ValueError('Expected one60s repeated and one600s record')
    pin_paths=[SOURCE/'README.md',SOURCE/'run_model/MC_off.ipynb',SOURCE/'retinatools/library.py',SOURCE/'stimuli/GunSpectra.txt',
               PREPARED/'GIN_SOURCE_INDEX.json',PREPARED/'GIN_RAW_SPIKES.npz',ROOT/'retina_sim/gin_source_binding_v1.py',Path(__file__)]
    arrays={};summaries=[]
    with np.load(PREPARED/'GIN_RAW_SPIKES.npz',allow_pickle=False) as z:
        for rec in records:
            path=ROOT/rec['source_relative_path'];pin_paths.append(path)
            if file_sha(path)!=rec['source_sha256']:raise ValueError('Target recording changed')
            metadata=roster[path.name.lower()];parsed=read_raw_spike_tables(path,metadata)[0]
            if parsed['raw_columns_sha256']!=rec['raw_table']['raw_columns_sha256']:raise ValueError('Raw source parsing changed')
            for trial,key in enumerate(rec['counts_keys']):
                counts=bin_live_trial(parsed['ticks'][trial],metadata['live_seconds'])
                np.testing.assert_array_equal(counts,z[key]);arrays[key]=counts
                arrays[f"{rec['record_id']}_trial{trial}_raw_ticks"]=parsed['ticks'][trial]
            if parsed['trials_in_table']==7:arrays[rec['record_id']+'_maintained_activity_raw_ticks']=parsed['ticks'][6]
            summaries.append(dict(record_id=rec['record_id'],source_path=rec['source_relative_path'],
                duration_s=metadata['live_seconds'],live_trials=len(rec['counts_keys']),
                counts_per_trial=[int(arrays[k].sum()) for k in rec['counts_keys']],
                stimulus_binding=('DIRECT_ARCHIVED_10MIN_MOVIE' if metadata['live_seconds']==600 else
                    'README_DECLARED_FIRST60S_OF10MIN; separate6x1_movie_absent; exact_decoded_prefix_not_verified'),
                declared_frame_zero_based=751,time_zero='source-corrected live-video start, not acquisition Video Starts',
                duplicate_tables_not_recounted=True,raw_pre_post_events_preserved=True))
    np.savez_compressed(out/'TARGET_COUNTS_AND_CONTEXT.npz',**arrays)
    movie=ROOT/index['movie']['path'];pin_paths.append(movie)
    if file_sha(movie)!=index['movie']['sha256']:raise ValueError('Movie source identity changed')
    video=cv2.VideoCapture(str(movie))
    if not video.isOpened():raise ValueError('Source movie decoder unavailable')
    codec=dict(backend=video.getBackendName(),fps=video.get(cv2.CAP_PROP_FPS),frames=video.get(cv2.CAP_PROP_FRAME_COUNT),
               width=video.get(cv2.CAP_PROP_FRAME_WIDTH),height=video.get(cv2.CAP_PROP_FRAME_HEIGHT),version=cv2.__version__)
    probes=[];preview=[]
    try:
        for n in range(766):
            ok,image=video.read()
            if not ok:raise ValueError('Source movie ended before declared live start')
            if n>=747:
                probes.append(dict(frame_zero_based=n,mean_BGR=image.mean(axis=(0,1)).tolist(),
                    spatial_std_BGR=image.std(axis=(0,1)).tolist(),decoded_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                preview.append(image[:,:,::-1].copy())
    finally:video.release()
    first_rgb=preview[751-747];lms=source_display_lms(first_rgb)
    ticks=np.arange(100,dtype=np.int64)*10
    np.savez_compressed(out/'SOURCE_INPUT_PREVIEW.npz',frame_zero_based=np.arange(747,766),decoded_RGB=np.stack(preview),
        first_declared_live_source_LMS=lms,time_bin_start_ticks_100us=ticks,
        first100ms_declared_frame_zero_based=frame_index_for_ticks(ticks))
    rf=read_author_fixed_rf_parameters(SOURCE/'run_model')[256]
    structure=read(ROOT/'reports/retinotopic_rf_revision/SPATIAL_POOLING_CANDIDATE_V1/STRUCTURE_RESULT.json')
    anchors=structure['routes'][0]['anchors'];ecc=roster['lss01256.txt']['eccentricity_deg']
    if not anchors['near']['patch_center_eccentricity_deg']<ecc<anchors['far']['patch_center_eccentricity_deg']:
        raise ValueError('Selected first target requires unapproved spatial extrapolation')
    binding=dict(schema='gin-real-cell-source-binding-v1',source_cell_key=cell,source_type='M off',mapped_model_type='parasol_RGC_OFF',
        type_qualification='Functional M/MC class; no morphology was measured',source_eccentricity_deg=ecc,
        source_azimuth_deg=None,source_animal_or_session_id=None,species='M. fascicularis; paper cohort, no per-cell animal map',
        source_archive_sha256=index['archive_sha256'],archive_receipt='data/source_packages/WP1_20261031/GIN_EXTRACTION_MANIFEST.json',
        movie_sha256=index['movie']['sha256'],movie_path=index['movie']['path'],records=summaries,author_fitted_RF=rf,
        RF_pixel_grid='linspace(-128,128,256), image-row y; source fitted location is not soma/pedicle or azimuth',
        RF_fractional_column=(rf['parameters']['xoff']+128)*255/256,RF_fractional_row=(rf['parameters']['yoff']+128)*255/256,
        field_width_deg=4.6,nominal_pixel_pitch_deg=4.6/256,documented_pupil_diameter_mm=4,
        geometry_source='https://pmc.ncbi.nlm.nih.gov/articles/PMC8998785/ Methods Stimuli and Cell identification; DOI10.1113/JP281200',
        geometry_scope='FOV and pupil are experiment cohort settings; source retinal azimuth not recovered',
        frame_clock=dict(declared_playback_hz=150,declared_first_live_frame_zero_based=751,source_ticks_per_s=10000,
                         spike_bins_ms=1,media_container_fps_used_for_model=False),
        source_LMS_units='archive relative source display values; not physical capture, td or Rstar/s',
        source_gamma_denominator=256,source_and_movie_pins_verified=True,source_clock_registered=False,
        source_RF_registration_accepted=False,azimuth_contract_accepted=False,display_to_native_contract_closed=False,
        network_identity_bound=False,compatible_background_bound=False,training_evaluation_split_frozen=False,
        local_network_identity=None,network_RF_coordinate_deg=None,background_identity=None,scene_to_native_identity=None,
        extrapolation_needed_at_selected_ecc=False,candidate_radius_anchors=anchors,
        full60s_forward_count_if_same_contract=1,shared_600s_prefix_admitted=False,
        archived60s_movie_present=False,sixty_second_repeats_have_direct_movie_byte_identity=False,
        FIT_READY=False,background_qualified=False,production_adopted=False)
    binding['neural_forward_blockers']=admission_blockers(binding);binding['neural_forward_admitted']=False
    write(out/'BINDING.json',binding);write(out/'DECODED_TIMING_EVIDENCE.json',dict(codec=codec,frames=probes,
        conclusion='Decoded frame750 already has content, README specifies751; no performance-selected shift applied',
        audio_sync_pulses_decoded=False,physical_sync_resolved=False))
    write(out/'SOURCE_IDENTITIES.json',{str(p.relative_to(ROOT)):file_sha(p) for p in pin_paths})
    with (out/'UNIT_TESTS.log').open('w',encoding='utf-8') as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_gin_source_binding_v1.py'))
    if not tests.wasSuccessful():raise ValueError('Source binding regression failed')
    try:require_neural_binding(binding)
    except ValueError as error:write(out/'NEURAL_GATE.json',dict(status='BLOCKED_SCIENTIFIC_INPUT_CONTRACT',reason=str(error),model_started=False))
    else:raise ValueError('Incomplete source binding was incorrectly admitted')
    write(out/'RESULT.json',dict(status='SOURCE_BINDING_PREPARED_NEURAL_NOT_ADMITTED',cell=cell,eccentricity_deg=ecc,
        raw_records=2,repeated_live_trials=6,tests_passed=tests.testsRun,wall_seconds=time.perf_counter()-began,
        network_run=False,fit_run=False,source_code_executed=False,flags_all_false=True))
    print(json.dumps(read(out/'RESULT.json'),ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    prepare(parser.parse_args().out)
