"""Compare two images on the existing frozen cone mosaic; stop at light capture."""
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageOps
from scipy.io import savemat
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from retina_platform.iset_frontend import digest, read_capture, source_signature
from retina_sim.atlas_population_network import load_population_record
from retina_sim.atlas_sources import verified_bytes
from retina_sim.contracts import canonical_hash
from retina_sim.isetbio_geometry import FixedISETConeMaster, cone_apertures


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aperture_support(full_xy, selected_rows):
    """Keep all six true neighbours of every displayed cone, without an all-pairs full-master matrix."""
    selected_rows = np.asarray(selected_rows, dtype=int)
    require(len(selected_rows) > 0 and len(np.unique(selected_rows)) == len(selected_rows), 'Unique selected cones required')
    require(np.all((selected_rows >= 0) & (selected_rows < len(full_xy))), 'Cone row outside frozen master')
    _, neighbours = cKDTree(full_xy).query(full_xy[selected_rows], k=7)
    rows = np.unique(neighbours)
    active = np.searchsorted(rows, selected_rows)
    require(np.array_equal(rows[active], selected_rows), 'Reduced support lost a selected cone')
    expected = cone_apertures(full_xy)[selected_rows]
    reduced = cone_apertures(full_xy[rows])[active]
    require(np.allclose(reduced, expected, rtol=1e-12, atol=1e-10), 'Full-master aperture neighbours were not preserved')
    return full_xy[rows], active, expected


def image_coordinates(shape, bounds, x, y):
    height, width = shape[:2]
    x0, x1, y0, y1 = bounds
    pitch = min((x1-x0)/width, (y1-y0)/height)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    return [(height-1)/2-(y-cy)/pitch, (width-1)/2+(x-cx)/pitch]


def sample_image(rgb, bounds, x, y, grey):
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and min(rgb.shape[:2]) >= 2, 'RGB image required')
    coords = image_coordinates(rgb.shape, bounds, x, y)
    return np.stack([map_coordinates(rgb[:, :, k].astype(float)/255, coords,
                     order=1, mode='constant', cval=grey, prefilter=False) for k in range(3)], axis=-1)


def load_geometry(root):
    platform = json.loads((root/'configurations/retina_platform_v1.json').read_bytes())
    binding = platform['atlas_target']['active_circuit_configuration']
    configuration = json.loads(verified_bytes(root/binding['path'], binding['sha256']))
    require('frozen_graph' in configuration, 'Demo requires the existing finite frozen cone record; no substitute geometry')
    record, graph = load_population_record(root, owner_bounds_mm=configuration['owner_bounds_mm'],
        halo_mm=configuration['halo_mm'], execution_profile='atlas_isetbio_geometry_v1', configuration_path=binding['path'])
    cones = record['cones']; coordinate = cones['coordinates']['rf']
    require(coordinate['unit'] == 'mm' and coordinate['frame'] == 'functional_visual_field', 'Unexpected cone coordinates')
    xy = np.asarray(coordinate['values'], dtype=float)
    ids, types = cones['global_cell_ids'], np.asarray(record['cone_subtypes'])
    require(xy.shape == (len(ids), 2) and len(types) == len(ids) and set(types) == {'L', 'M', 'S'}, 'Frozen cone identity/type mismatch')
    master = FixedISETConeMaster(root, record['atlas']['cone_master'])
    lookup = {key: i for i, key in enumerate(master.source_keys)}
    indices = np.array([lookup[key] for key in cones['source_keys']])
    require(np.array_equal(xy, master.positions[indices]), 'Cone coordinates differ from the immutable master')
    support, active, apertures = aperture_support(master.positions, indices)
    provenance = dict(configuration=binding, graph=graph, master=record['atlas']['cone_master'],
        master_schema=master.manifest['schema'], density_species=master.manifest['density_species'],
        type_provenance=record['cone_type_provenance'], query_cones=len(ids),
        reference_id=record['atlas']['reference_id'],
        selection='All cones in the existing finite graph query record; not a newly generated mosaic or the downstream dynamical subset',
        aperture_support=dict(full_master_count=len(master.positions), reduced_count=len(support),
                              all_selected_six_neighbours_preserved=True))
    return xy, ids, types, support, active, apertures, provenance, configuration['halo_mm']


def load_central_geometry(root):
    """Crop the existing macaque master; retain its global IDs and subtype domains."""
    from retina_sim.source_atlas import SourceDensityAtlas
    from retina_sim.atlas_s_partition import EffectiveMulattaSPartition
    atlas = SourceDensityAtlas(root, profile_id='retina_atlas_isetbio_source_stop_v1')
    master = atlas.cone_master
    require(master.bounds[0] <= -.446 and master.bounds[1] >= .446 and
            master.bounds[2] <= -.446 and master.bounds[3] >= .446, 'Central disk exceeds complete master support')
    selected = np.flatnonzero(np.linalg.norm(master.positions, axis=1) <= .446)
    xy = master.positions[selected]
    table = atlas._table('cones', tuple(master.source_keys[i] for i in selected), xy)
    labels, type_provenance = EffectiveMulattaSPartition(atlas).assign(table)
    apertures = cone_apertures(master.positions)[selected]
    with np.load(root/master.manifest['payload']['path'], allow_pickle=False) as payload:
        require(np.allclose(apertures, payload['aperture_diameters_um'][selected], rtol=1e-12, atol=1e-10),
                'Apertures differ from the frozen full-master values')
    _, local_ids, local_types, *_ = load_geometry(root)
    lookup = dict(zip(table.global_cell_ids, labels))
    overlap = [i for i,gid in enumerate(local_ids) if gid in lookup]
    require(overlap and all(lookup[local_ids[i]] == local_types[i] for i in overlap), 'Existing overlap subtype assignments changed')
    geometry = dict(master=dict(atlas.config['cone_master']), master_schema=master.manifest['schema'],
        density_species=master.manifest['density_species'], reference_id=atlas.config['reference_id'],
        selection='Existing frozen master cone centers in a fovea-centered radius 2 degree disk; no regeneration',
        center_deg=[0.,0.], radius_deg=2., diameter_deg=4., query_cones=len(xy),
        type_provenance=type_provenance, previous_local_overlap_cones=len(overlap),
        previous_local_overlap_types_unchanged=True, aperture_rule='0.79 times full-master median six-neighbor spacing')
    geometry['eccentricity_range_deg'] = [float(np.linalg.norm(xy,axis=1).min()/.223),float(np.linalg.norm(xy,axis=1).max()/.223)]
    geometry['recipe_hashes'] = central_recipe_hashes(root)
    return xy, list(table.global_cell_ids), np.asarray(labels), apertures, geometry


def central_recipe_hashes(root):
    paths=('retina_sim/geometry.py','retina_sim/source_atlas.py','retina_sim/atlas_s_partition.py',
           'configurations/retina_atlas_isetbio_source_stop_v1.json',
           'configurations/deMonasterio1985_mulatta_effective_S_v1.json',
           'sources/spatial/demonasterio1985_Scone_reference.json')
    return dict(numpy_version=np.__version__,files={name:digest(root/name) for name in paths})


def load_saved_central_geometry(root, directory):
    """Reuse this demo's saved label domains only with unchanged source rules and master."""
    from retina_sim.source_atlas import SourceDensityAtlas
    geometry=json.loads((directory/'geometry.json').read_bytes())
    require(geometry.get('recipe_hashes')==central_recipe_hashes(root),'Saved geometry source recipe changed')
    atlas=SourceDensityAtlas(root,profile_id='retina_atlas_isetbio_source_stop_v1')
    require(geometry['master']==dict(atlas.config['cone_master']) and geometry['radius_deg']==2. and
            geometry['center_deg']==[0.,0.], 'Saved central geometry uses a different master/region')
    provenance=geometry['type_provenance']
    require(provenance['implementation_sha256']==digest(root/'retina_sim/atlas_s_partition.py') and
            provenance['profile_sha256']==digest(root/'configurations/deMonasterio1985_mulatta_effective_S_v1.json'),
            'Saved subtype rules changed; regenerate the central geometry')
    with np.load(directory/'geometry.npz',allow_pickle=False) as data:
        xy,ids,types,apertures=(data[key].copy() for key in ('xy_mm','cell_ids','types','apertures_um'))
    master=atlas.cone_master; selected=np.flatnonzero(np.linalg.norm(master.positions,axis=1)<=.446)
    table=atlas._table('cones',tuple(master.source_keys[i] for i in selected),master.positions[selected])
    require(np.array_equal(xy,master.positions[selected]) and np.array_equal(ids,table.global_cell_ids) and
            types.shape==(len(ids),) and set(types)=={'L','M','S'}, 'Saved cone identity/selection differs')
    require(np.allclose(apertures,cone_apertures(master.positions)[selected],rtol=1e-12,atol=1e-10),
            'Saved apertures differ from full-master support')
    geometry=dict(geometry,reused_geometry=dict(directory=str(directory.resolve()),
        npz_sha256=digest(directory/'geometry.npz'),metadata_sha256=digest(directory/'geometry.json')))
    return xy,ids.tolist(),types,apertures,geometry


def registration_check(xy, types, rates, background, shape, bounds):
    row, col = image_coordinates(shape, bounds, xy[:, 0], xy[:, 1])
    u, v = col/(shape[1]-1), row/(shape[0]-1)
    relative = rates/background
    horizontal = (types == 'L') & (v > .3) & (v < .8)
    vertical = (types == 'M') & (u > .35) & (u < .65)
    selections = [horizontal & (u > .1) & (u < .3), horizontal & (u > .7) & (u < .9),
                  vertical & (v > .25) & (v < .4), vertical & (v > .65) & (v < .85)]
    require(all(np.count_nonzero(mask) >= 3 for mask in selections), 'Insufficient cones for image-axis registration check')
    means = [float(relative[mask].mean()) for mask in selections]
    require(means[1] > means[0] and means[3] > means[2], 'Image/capture x or y orientation differs; do not publish the comparison')
    return dict(L_right_over_grey=means[1], L_left_over_grey=means[0],
                M_bottom_over_grey=means[3], M_top_over_grey=means[2], passed=True,
                normalization_use='Background ratios only for this axis check; plotted rates are absolute and unnormalized')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory; never overwritten')
    parser.add_argument('--natural', type=Path, default=ROOT/'vendor_pinned/isetcam/data/images/rgb/FruitMCC_6500.png')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Output exists; choose a fresh directory to preserve earlier figures')
    profile, sources = source_signature(ROOT)
    source_paths = [Path(__file__), ROOT/'tools/export_cone_mosaic_demo.m', ROOT/'tools/plot_cone_aperture_check.m',
                    ROOT/'retina_sim/isetbio_geometry.py', ROOT/'retina_sim/atlas_population_network.py',
                    ROOT/'retina_sim/atlas_sources.py']
    sources.update({str(path): digest(path) for path in source_paths})
    xy, ids, types, support, active, apertures, geometry, halo = load_geometry(ROOT)
    paths = [(ROOT/'examples/platform_media/synthetic_rgb.png').resolve(), args.natural.resolve()]
    images, inputs = [], []
    for name, path in zip(('synthetic', 'natural'), paths):
        require(path.is_file() and path.stat().st_size <= 16*1024**2, 'Missing or oversized image: '+str(path))
        image_sha = digest(path)
        with Image.open(path) as image:
            require(image.width*image.height <= 16*1024**2, 'Image pixel budget exceeded')
            images.append(np.asarray(ImageOps.exif_transpose(image).convert('RGB')))
        require(digest(path) == image_sha, 'Input image changed during decoding')
        inputs.append(dict(name=name, source_path=str(path), sha256=image_sha,
                           size_pixels=list(images[-1].shape[1::-1])))
    natural_source = args.natural.parent/'source.json'
    if paths[1] == (ROOT/'vendor_pinned/isetcam/data/images/rgb/FruitMCC_6500.png').resolve():
        inputs[1]['attribution'] = dict(title='FruitMCC_6500', repository='https://github.com/ISET/isetcam',
            commit=profile['isetcam_commit'], path='data/images/rgb/FruitMCC_6500.png',
            repository_license='MIT', repository_license_sha256=digest(ROOT/'vendor_pinned/isetcam/LICENSE'),
            image_specific_credit='No separate image-specific author/permission statement found beside the bundled sample; retain repository license and verify image-specific rights for formal external publication')
    elif natural_source.is_file():
        attribution = json.loads(natural_source.read_bytes())
        require(attribution['sha256'] == inputs[1]['sha256'], 'Natural image provenance hash differs')
        inputs[1]['attribution'] = attribution
    else:
        inputs[1]['attribution'] = {'status': 'User-supplied image; publication rights must be confirmed'}
    margin = float(apertures.max())/2000 + .001
    bounds = np.array([xy[:, 0].min()-margin, xy[:, 0].max()+margin,
                       xy[:, 1].min()-margin, xy[:, 1].max()+margin])
    center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2])
    extent = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2]])+2*halo
    cols, rows = np.ceil(extent/(profile['scene_sampling_um']*.001)).astype(int)+1
    require(max(rows, cols) <= profile['maximum_scene_samples_per_axis'], 'Optical scene exceeds existing grid budget')
    pitch = max(extent[0]/(cols-1), extent[1]/(rows-1))
    x = (np.arange(cols)-(cols-1)/2)*pitch+center[0]
    y = ((rows-1)/2-np.arange(rows))*pitch+center[1]
    xx, yy = np.meshgrid(x, y)
    u = profile['reference_linear_value']
    grey = 12.92*u if u <= .0031308 else 1.055*u**(1/2.4)-.055
    frames = np.stack([sample_image(rgb, bounds, xx, yy, grey) for rgb in images], axis=3)
    dt = .010
    edges = np.arange(3, dtype=float)*dt
    request = canonical_hash(dict(inputs=inputs, geometry=geometry, profile=profile, bounds=bounds,
                                  xy=xy, types=types.tolist(), dt=dt, frame_registration='top_is_positive_project_y'))
    output.mkdir(parents=True)
    for item, path in zip(inputs, paths):
        shutil.copy2(path, output/(item['name']+'_source'+path.suffix))
    shutil.copy2(ROOT/'vendor_pinned/isetcam/LICENSE', output/'ISETCam_MIT_LICENSE.txt')
    source_mat = output/'input.mat'
    savemat(source_mat, dict(rgb_frames=frames.astype(np.float32), window_edges_seconds=edges,
        frame_indices=np.array([1., 2.]), reference_rgb=np.full(3, grey), native_apertures_um=cone_apertures(support)[active], support_is_mm=support,
        active_indices=active.astype(float)+1, is_mm=xy,
        types=np.array([{'L': 1., 'M': 2., 'S': 3.}[kind] for kind in types]), image_center_deg=center/.223,
        plot_bounds_mm=bounds, plot_x_mm=x, plot_y_mm=y), do_compression=True)
    focal_length_m = .223e-3/np.tan(np.pi/180)
    export = dict(profile=profile, input_mat=str(source_mat), patch_fov_deg=float(np.rad2deg(2*np.arctan(cols*pitch*.001/(2*focal_length_m)))),
                  request_signature=request, background_signature=canonical_hash(dict(profile=profile, geometry=geometry, xy=xy)))
    config_path = output/'export.json'
    config_path.write_text(json.dumps(export, allow_nan=False, indent=2), encoding='utf-8')
    expression = "addpath('"+str(ROOT/'tools').replace("'", "''")+"');export_cone_mosaic_demo('"+str(config_path).replace("'", "''")+"','"+str(output/'capture.mat').replace("'", "''")+"');"
    preferences = output/'matlab_preferences'; preferences.mkdir()
    temporary = output/'matlab_temp'; temporary.mkdir()
    environment = dict(os.environ, MATLAB_PREFDIR=str(preferences), TEMP=str(temporary), TMP=str(temporary))
    print(f'Capturing {len(ids)} frozen cones, two static images; no downstream circuit.', flush=True)
    with (output/'matlab_stdout.log').open('wb') as stdout, (output/'matlab_stderr.log').open('wb') as stderr:
        result = subprocess.run([profile['matlab_executable'], '-batch', expression], env=environment,
            stdout=stdout, stderr=stderr, timeout=profile['maximum_export_wall_seconds']+120,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0, 'MATLAB capture/render failed; preserve logs in '+str(output))
    raw, meta = read_capture(output/'capture.mat', request)
    counts = raw['expected_counts']
    require(counts.shape == (2, len(ids)) and meta['phototransduction_applied'] is False and meta['photon_noise'] == 'none', 'Capture scope changed')
    require(np.array_equal(raw['window_edges_seconds'].reshape(-1), edges), 'Actual capture integration windows differ')
    require(np.allclose(raw['diameters_um'].reshape(-1), apertures, rtol=1e-11, atol=1e-9), 'Actual ISET aperture differs from full-master aperture')
    rates = counts/dt
    rendering = json.loads((output/'render_metadata.json').read_text(encoding='utf-8'))
    require(rendering['color_range'][0] == 0 and rendering['color_range'][1] >= float(rates.max()), 'Rendered color scale clips physical rates')
    axis_check = registration_check(xy, types, rates[0], raw['background_counts'].reshape(-1)/dt, images[0].shape, bounds)
    require(all((output/(name+suffix)).is_file() for name in ('synthetic', 'natural') for suffix in ('_comparison.png', '_comparison.pdf')), 'Missing rendered comparisons')
    with (output/'cone_capture.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cell_id', 'type', 'project_rf_x_mm', 'project_rf_y_mm', 'model_aperture_diameter_um',
                         'synthetic_expected_Rstar_per_10ms', 'natural_expected_Rstar_per_10ms',
                         'synthetic_expected_Rstar_per_cone_s', 'natural_expected_Rstar_per_cone_s'])
        for i, cell_id in enumerate(ids):
            writer.writerow([cell_id, types[i], *xy[i], apertures[i], *counts[:, i], *rates[:, i]])
    require(all(digest(Path(path)) == expected for path, expected in sources.items()), 'Source changed during capture; retain outputs but do not certify this run')
    require(all(digest(path) == item['sha256'] for path, item in zip(paths, inputs)), 'Input changed during capture')
    manifest = dict(schema='cone_mosaic_capture_demo_v1', inputs=inputs, geometry=geometry, source_hashes=sources,
        cell_count=len(ids), type_counts={kind: int(np.sum(types == kind)) for kind in ('L', 'M', 'S')},
        cone_identity_sha256=canonical_hash(dict(ids=ids, xy=xy, types=types.tolist())),
        integration_seconds=dt, value_semantics='expected', displayed_unit='R*/cone/s', stored_unit='R*/10ms',
        color_range=rendering['color_range'], data_max_rate_rstar_per_cone_s=float(rates.max()), shared_color_range_across_inputs=True,
        per_image_or_per_type_normalization=False, registration_check=axis_check,
        plot_bounds_mm=bounds.tolist(), field_size_degrees=((bounds[[1, 3]]-bounds[[0, 2]])/.223).tolist(),
        fixed_gaze=True, pixel_mapping='Aspect-preserving fit; fixed reference-grey letterbox and optical halo; input top maps +project y',
        figure_language='English', figure_subtitles=False, figure_captions=False,
        raw_capture=meta, downstream_circuit_run=False, phototransduction_run=False,
        physiological_validation=False, simulation_stage='Only display/optics/cone photopigment capture',
        runtime=dict(python=sys.version, numpy=np.__version__, matlab=meta['matlab_version']),
        source_unchanged_during_capture=True,
        outputs={p.name: digest(p) for p in output.iterdir() if p.suffix in ('.png', '.pdf', '.csv', '.mat')})
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (output/'README.txt').write_text(
        '原图—L/M/S 类型—逐细胞光捕获响应对照\n\n'
        f'两版共用现有冻结查询域的 {len(ids)} 个 cone；位置、ID、类型未重新生成。\n'
        '这是完整局部查询域的几何预览，不等同于后续电路筛选的动态 cone 子集，也不是全视网膜覆盖。\n'
        '合成图：项目既有 RGB 色块与渐变。自然图：本次 --natural 所选文件；原始尺寸、来源及已有裁剪记录见 manifest 的 inputs。\n'
        '随附 ISETCam MIT 仓库许可证。该照片旁未发现独立摄影者/图像许可声明，正式对外发表前应确认单幅照片的素材权利。\n'
        '输入等比例放入同一视野，以固定参考灰补齐；图 A 显示实际用于光学计算的输入，未拉伸原图。\n'
        'B 的颜色仅表示 L/M/S 类型；圆盘使用模型采光孔径，不是实测细胞胞体大小。\n'
        'C 显示绝对期望光异构化率 R*/cone/s，两图共享从零开始的色标，无逐图/逐类型归一化。\n'
        '原始数据为每个 10 ms 窗口的期望 R* 数，除以 0.010 s 得到速率；非整数是期望值，未加入光子噪声。\n'
        '几何：现有 Macaca nemestrina 密度约束的 ISET 数值格点；L/M/S 为现有有效分配，S 路径含 Macaca mulatta 跨物种借用。\n'
        '光学：沿用 LCD-Apple gamma/SPD、人眼 MW 光学、3 mm 瞳孔、人眼 lens/macular 及既有光色素设定。\n'
        '固定参考灰标定为 100 Td；这是模型内统一辐亮度增益，不是对照片原始场景光谱/照度的恢复。\n'
        '位置按既有 RF=IS 项目近似注册，角度换算 223 um/deg；有限光学背景支持是数值近似。\n'
        '孔径为完整冻结格点六近邻间距中位数的 0.79 倍，已核对实际 ISET 输出；裁剪支持没有改变孔径。按用户选择，不引入外部 S-cone 直径比例。\n'
        '空间积分使用每个 cone 的实际 ISETBio Pillbox 核，仅合并逐元素完全相同的离散核，不再以全体平均孔径代替。\n'
        '外节衰减按 L/M/S 各自吸收谱和已有光密度计算，替代原统一 L-cone 标量修正；仍保留 ISETBio 波长均值近似。\n'
        'ISETBio 人类 CVRL 中央凹外节长度为 L/M 35.5 um、S 33.725 um；按已有光密度反推比密度，避免重复计入 S 外节缩短。\n'
        '所有图内文字为英文，无副标题或图注；模型假设与证据说明保留在独立文件中。\n'
        '未运行光转导、HC/BC/AC/RGC 电路、适应过程、训练或拟合；不能称为实测或已标定的猕猴生理响应。\n'
        '两个静态案例采用相同 10 ms 窗口；MAT 文件两行是两个独立案例，不用于解释时间动态。\n'
        '源图副本、输入采样、原始 capture.mat、逐细胞 CSV、图像/PDF、模型/图源哈希与 MATLAB 日志均保留。\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), cells=len(ids), type_counts=manifest['type_counts'], registration=axis_check), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
