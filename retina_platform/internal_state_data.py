"""Read saved 69#6 states without constructing or advancing a retinal model."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify(path, expected):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Saved artifact checksum differs: {path}")


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"dictionary"}:
            return {_decode(k): _decode(v) for k, v in value["dictionary"]}
        if set(value) == {"tuple"}:
            return tuple(_decode(v) for v in value["tuple"])
    return value


class _Bundle:
    def __init__(self, path, identity, fields):
        self.path = Path(path)
        manifest = _read(self.path / "manifest.json")
        if manifest["identity"] != identity:
            raise ValueError("Saved state belongs to a different model background")
        self.files = manifest["files"]
        self._verified = set()
        self.payload = {
            k: _decode(v) for k, v in manifest["payload"]["dictionary"] if k in fields
        }

    def array(self, ref):
        name = ref["array"]
        if Path(name).name != name or name not in self.files:
            raise ValueError("Invalid saved array reference")
        if name not in self._verified:
            _verify(self.path / name, self.files[name]["sha256"])
            self._verified.add(name)
        arr = np.load(self.path / name, allow_pickle=False)
        entry = self.files[name]
        if list(arr.shape) != entry["shape"] or arr.dtype.str != entry["dtype"]:
            raise ValueError("Saved array shape or unit representation changed")
        return arr


class SavedNetworkData:
    """Small metadata index and on-demand reads of existing numeric artifacts.

    A snapshot is available only at its actual timestamp. Group responses are
    indexed by group row; individual optical captures use their own identities.
    """

    def __init__(self, project_root, result=None):
        from .internal_state_geometry import SavedGeometry

        self.project_root = Path(project_root).resolve()
        report = self.project_root / "reports/GIN_69_6_HC2_BC1_SUPPORT_V2"
        selected = Path(result).resolve() if result is not None else report / "current_ac_source"
        if selected.is_file():
            selected = selected.parent
        if selected.name == 'internal_states' and (selected.parent / 'BACKGROUND.json').exists():
            selected = selected.parent
        if (selected / "current_ac_source/BACKGROUND.json").exists():
            selected = selected / "current_ac_source"
        self.result = selected
        self.source = self.result
        self.report = selected.parent
        background = _read(selected / "BACKGROUND.json")
        neural = _read(selected / "NEURAL_RESULT.json")
        if neural["status"] != "COMPLETE" or neural["model_identity"] != background["model_identity"]:
            raise ValueError("A complete matching saved neural result is required")
        self.identity = background["model_identity"]
        self.background_identity = background["identity"]
        self.duration = float(neural["source_seconds"])
        metrics = _read(selected / "METRICS.json")
        receipt = _read(selected / "PREFIX_45s.json")
        if (receipt["background_identity"] != self.background_identity
                or metrics["neural"]["model_identity"] != self.identity
                or receipt["source_cell"] != "69#6" or receipt["target_group"] != 7730):
            raise ValueError("Saved result identity chain differs")
        for relative, expected in (("PREFIX_45s.npz", receipt["sha256"]),
                ("readout/FIXED_DEVELOPMENT_PREDICTIONS.npz", metrics["predictions_sha256"]),
                ("readout/MODEL_FROZEN.json", metrics["frozen_model_sha256"]),
                ("seed/manifest.json", background["seed_manifest_sha256"]),
                ("CHECKPOINT_450000/manifest.json", metrics["verification"]["final_checkpoint_manifest_sha256"]),
                ("CHECKPOINT_50000/manifest.json", neural["checks"]["checkpoint_manifest_sha256"])):
            _verify(selected / relative, expected)
        self.limitations = [
            "连续 0–45 s 神经轨迹仅保存 69#6 的膜电位、电导与原期望率；其他内部变量仅快照。",
            "Cone/HC 原生状态时刻为 0、5.0005、45 s；BC/AC/AII 为 0、5、45 s。",
            "Cone release 仅保存输入桥接所选组的 5、45 s 快照；未记录组和时刻显示缺失。",
            "分组与功能锚点不代表多个独立细胞响应；原始成员坐标单独保留。",
            "冻结 GLM 保存的是自由历史平均预测；未保存单条 GLM Spike 路径。",
            "光学为此实验冻结的 Gaussian 孔径及借用的人 Cone 晶格；原生电压不是 mV。",
        ]
        self._geometry = SavedGeometry(self.project_root, selected)
        self.geometry = self._geometry.geometry
        self.populations = self._geometry.populations
        self.target = self._geometry.target
        self.available_h1_rows = [13, 2781]
        self.rgcs = []
        for pop, row in self._geometry.rgcs:
            g = self.geometry[pop]
            target = (pop, row) == self.target
            eccentricity = 3.88 if target else float(np.linalg.norm(g['xy_mm'][row]) / .223)
            self.rgcs.append(dict(population=pop, row=int(row), id=str(g["ids"][row]),
                type=pop, label=("69#6 · OFF-parasol · 3.88°" if target else f"{pop} · {g['ids'][row]} · 模型锚点 {eccentricity:.3f}°"),
                eccentricity_deg=eccentricity, eccentricity_role='记录偏心度' if target else '模型 RF 锚点换算', is_target=target))
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._seed = _Bundle(selected / "seed", self.background_identity, {"states", "native_inputs"})
        self._snapshots = {}
        self._variables = {}
        self._events = {}
        self._index_snapshots()
        self.continuous_window = None
        self.window_manifest = None
        self._window_fields = {}
        self._window_checked = set()
        self._index_continuous_window()
        self._optics = _read(self.report / "formal_optics/MANIFEST.json")
        self._frame_times = np.asarray([f["time_frame_start_seconds"] for f in self._optics["frames"]])
        self._optical_geometry = None
        self.snapshot_times = sorted({p[0] for points in self._snapshots.values() for p in points})

    def network(self, target=None):
        return self._geometry.network(target or self.target)

    def h1_inputs(self, row=None):
        return self._geometry.h1_inputs(row)

    def _index_continuous_window(self):
        path = self.result / 'internal_states/manifest.json'
        if not path.exists():
            return
        manifest = _read(path)
        if manifest.get('status') != 'COMPLETE':
            self.limitations.append('短窗口记录尚未完成；继续显示原保存结果。')
            return
        if (manifest['model_identity'] != self.identity
                or manifest['background_identity'] != self.background_identity):
            raise ValueError('Short-window states belong to a different model')
        _verify(self.result / 'CHECKPOINT_50000/manifest.json', manifest['checkpoint_manifest_sha256'])
        filename = manifest['responses_file']
        if Path(filename).name != filename:
            raise ValueError('Invalid short-window array path')
        self._window_file = 'internal_states/' + filename
        _verify(self.result / self._window_file, manifest['responses_sha256'])
        time = self._npz(self._window_file, manifest['time_array'])
        start, stop = manifest['start_seconds'], manifest['stop_seconds']
        if (time.ndim != 1 or not 2 <= len(time) <= 10001 or not np.isfinite(time).all()
                or not 0 <= start < stop <= self.duration or np.any(np.diff(time) <= 0)
                or abs(time[0] - start) > 1e-12 or abs(time[-1] - stop) > 1e-12
                or not np.allclose(np.diff(time), manifest['sample_seconds'], rtol=0, atol=1e-12)):
            raise ValueError('Invalid saved short-window time grid')
        for pop, meta in manifest['populations'].items():
            g = self.geometry[pop]
            if (list(map(str, meta['ids'])) != list(map(str, g['ids']))
                    or meta['members'] != g['members'].tolist()
                    or meta['coordinate_role'] != g['coordinate_role']):
                raise ValueError('Short-window member identities or coordinate roles differ: ' + pop)
        for token, meta in manifest['fields'].items():
            pop, key = meta['population'], meta['key']
            if pop not in manifest['populations'] or meta['array'] != token:
                raise ValueError('Unbound short-window response field')
            rows = np.asarray(meta['rows'], dtype=int)
            if (rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows)
                    or np.any(rows < 0) or np.any(rows >= len(self.geometry[pop]['ids']))):
                raise ValueError('Invalid short-window response row mapping')
            existing = self.variables(pop).get(key)
            if existing and existing['unit'] != meta['unit']:
                raise ValueError('Short-window field unit changed: ' + token)
            self._window_fields[pop, key] = dict(meta, rows=rows)
            self._variables.setdefault(pop, {})[key] = dict(
                unit=meta['unit'], label=existing['label'] if existing else meta['label'],
                recording=existing['recording'] if existing and existing['recording'] == 'continuous' else 'windowed',
                continuous_window=[start, stop], source_kind=meta['source_kind'])
        self._window_time = time
        self.continuous_window = (float(start), float(stop))
        self.window_manifest = manifest
        self.limitations[:3] = [
            f'{start:g}–{stop:g} s 已从原检查点重放并补存跨层连续状态；窗口外仍只有原记录。',
            '短窗按实际 1 ms 节点显示；原 RGC 保留其完整 100 μs 轨迹。',
            '短窗 Cone release 为全部组的原生输出；窗外 release 是原桥接缓存的部分组快照。',
        ]

    def _window_values(self, pop, key):
        meta = self._window_fields[pop, key]
        values = self._npz(self._window_file, meta['array'])
        if (pop, key) not in self._window_checked:
            if values.shape != (len(meta['rows']), len(self._window_time)) or not np.isfinite(values).all():
                raise ValueError('Short-window response shape or values differ: ' + meta['array'])
            self._window_checked.add((pop, key))
        return values

    def _remember(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
            self._cache_bytes += self._cache[key].nbytes
            while len(self._cache) > 1 and self._cache_bytes > 96 * 1024**2:
                _, removed = self._cache.popitem(last=False)
                self._cache_bytes -= removed.nbytes
        self._cache.move_to_end(key)
        return self._cache[key]

    def _npz(self, filename, key):
        path = self.result / filename
        def read():
            with np.load(path, allow_pickle=False) as z:
                return z[key]
        return self._remember((str(path), key), read)

    def _add(self, pop, key, time, bundle, ref, unit, label, indices=None):
        self._snapshots.setdefault((pop, key), []).append((float(time), bundle, ref, indices))
        self._variables.setdefault(pop, {})[key] = dict(unit=unit, label=label, recording="snapshot")

    def _index_snapshots(self):
        for key, ref in self._seed.payload["states"].items():
            self._add_inner(key, 0., self._seed, ref)
        for key in ("Vis", "Vbar"):
            self._add("cones", "V" if key == "Vis" else key, 0., self._seed,
                self._seed.payload["states"]["cones", "reduced_native", key],
                "vH_native_voltage", "Cone 原生电压" if key == "Vis" else "Cone Vbar 原生状态")
        self._h1_rows = np.flatnonzero(np.asarray(self.geometry["HC"]["subtypes"]) == "H1")
        self._h2_rows = np.flatnonzero(np.asarray(self.geometry["HC"]["subtypes"]) == "H2")
        if len(self._h1_rows) != 325 or len(self._h2_rows) != 126:
            raise ValueError("Saved HC state and H1/H2 row mapping differ")
        for field, rows in (("h1", self._h1_rows), ("h2", self._h2_rows)):
            self._add("HC", "Vh", 0., self._seed,
                self._seed.payload["states"]["HC", "h1_h2_effective_v1", field],
                "vH_native_voltage", "H1/H2 原生反馈状态", ("scatter", rows))
        for field in ("w", "z2"):
            self._add("HC", field, 0., self._seed,
                self._seed.payload["states"]["HC", "h1_h2_effective_v1", field],
                "vH_native_voltage", f"H1 {field} 状态", ("scatter", self._h1_rows))
        for name in ("CHECKPOINT_50000", "CHECKPOINT_450000"):
            folder = self.result / name
            if not (folder / "manifest.json").exists():
                continue
            bundle = _Bundle(folder, self.background_identity, {"inner", "native", "controller", "readout_cursor"})
            ctrl = bundle.payload["controller"]
            time = float(ctrl["inner_time"])
            for key, ref in bundle.payload["inner"].items():
                self._add_inner(key, time, bundle, ref)
            point = bundle.payload["native"]["point"]
            layout = bundle.payload["native"]["intervals"][-1]
            offsets = np.r_[0, np.cumsum(layout["state_sizes"])]
            for i, field in enumerate(layout["state_names"]):
                if field not in ("Vis", "Vbar", "Vh", "w", "z2"):
                    continue
                pop = "cones" if field in ("Vis", "Vbar") else "HC"
                key = "V" if field == "Vis" else field
                index = slice(int(offsets[i]), int(offsets[i + 1]))
                if field in ("w", "z2"):
                    index = ("native_scatter", index, self._h1_rows)
                self._add(pop, key, point["time_seconds"], bundle, point["y"],
                    "vH_native_voltage", f"{'Cone' if pop == 'cones' else 'H1/H2'} {field} 原生状态", index)
            release = bundle.payload["readout_cursor"]
            self._add("cones", "release", release["release_cache_time"], bundle,
                release["release_cache"], "dimensionless effective release", "Cone 模型有效释放",
                ("scatter", self._geometry.cone_release_rows))
            if time == self.duration:
                self._events = ctrl["event_log"]
            # Large solver counters and unneeded native polynomial histories are
            # not retained by the viewer; array descriptors above remain valid.
            bundle.payload = {}
        for pop in self.geometry:
            if pop in self._events:
                self._variables.setdefault(pop, {})["spike"] = dict(unit="event", label="原 RGC 保存的 Spike", recording="continuous")
        self._variables.setdefault("cones", {})["native_input"] = dict(
            unit="reference-equivalent native drive", label="Cone 已保存输入（参考等效工程量）", recording="continuous")
        for key, unit, label in (("V", "V", "原 RGC 膜电位"), ("gE", "S", "原 RGC 兴奋电导"),
                ("gI", "S", "原 RGC 抑制电导"), ("expected_rate_Hz", "Hz", "原 RGC 期望放电率"),
                ("glm_rate_Hz", "Hz", "冻结 GLM 自由预测期望率")):
            self._variables[self.target[0]][key] = dict(unit=unit, label=label,
                recording="continuous", availability="仅 69#6 连续；其他 RGC 仅已有快照")

    def _add_inner(self, state, time, bundle, ref):
        pop, mechanism, field = state
        if pop not in self.geometry or pop in ("cones", "HC"):
            return
        if field == "V":
            key, unit, label = "V", "V", "膜电位"
        elif field in ("terminal_release", "release"):
            key, unit, label = "release", "dimensionless effective release", "模型有效释放"
        elif field in ("s", "h", "A"):
            key, unit, label = field, "dimensionless", {"s": "输出门控", "h": "抑制状态", "A": "适应状态"}[field]
        else:
            return
        self._add(pop, key, time, bundle, ref, unit, label)

    def variables(self, pop):
        return self._variables.get(pop, {})

    def _snapshot_values(self, pop, key, time):
        values = np.full(len(self.geometry[pop]["ids"]), np.nan)
        for timestamp, bundle, ref, indices in self._snapshots.get((pop, key), []):
            if not np.isclose(timestamp, time, rtol=0, atol=1e-9):
                continue
            arr = np.asarray(bundle.array(ref))
            if isinstance(indices, tuple) and indices[0] == "scatter":
                values[indices[1]] = arr
            elif isinstance(indices, tuple) and indices[0] == "native_scatter":
                values[indices[2]] = arr[indices[1]]
            else:
                selected = arr if indices is None else arr[indices]
                if selected.shape != values.shape:
                    raise ValueError(f"Saved {pop}/{key} and geometry row counts differ")
                values[:] = selected
        return values

    @staticmethod
    def _subsample(times, values, start, stop, max_points, preserve_extrema=False):
        lo, hi = np.searchsorted(times, [start, stop], side="left")
        if hi < len(times) and times[hi] == stop:
            hi += 1
        indices = np.arange(lo, hi)
        if len(indices) > max_points:
            if preserve_extrema and max_points >= 4:
                edges = np.linspace(lo, hi, (max_points - 2) // 2 + 1, dtype=int)
                selected = [lo, hi - 1]
                for a, b in zip(edges[:-1], edges[1:]):
                    segment = values[a:b]
                    selected.extend((a + int(np.argmin(segment)), a + int(np.argmax(segment))))
                indices = np.unique(selected)
            else:
                indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=int)]
        return times[indices], values[indices]

    def series(self, pop, row, key, start=0., stop=45., max_points=2000):
        if pop not in self.geometry or not 0 <= row < len(self.geometry[pop]["ids"]):
            raise IndexError("Unknown saved population row")
        info = self.variables(pop).get(key, dict(unit="", label=key, recording="snapshot"))
        times = values = np.array([], dtype=float)
        recording = "snapshot"
        count = None
        if (pop, row) == self.target and key in ("V", "gE", "gI", "expected_rate_Hz", "glm_rate_Hz"):
            recording = "continuous"
            if key in ("V", "gE", "gI"):
                times = self._npz("PREFIX_45s.npz", "time_source_s")
                values = self._npz("PREFIX_45s.npz", key)
            else:
                times = self._npz("readout/FIXED_DEVELOPMENT_PREDICTIONS.npz", "time_bin_start_seconds")
                values = (self._npz("readout/FIXED_DEVELOPMENT_PREDICTIONS.npz", "probability") * 1000
                    if key == "glm_rate_Hz" else self._npz("PREFIX_45s.npz", "expected_rate_base_Hz"))
        elif key == "spike":
            times = np.asarray([e["time_seconds"] for e in self._events.get(pop, [])
                if e["cell_index"] == row and 0 <= e["time_seconds"] < self.duration])
            values = np.ones(len(times))
            recording = "events"
        elif pop == "cones" and key == "native_input":
            indices = np.flatnonzero((self._frame_times >= start) & (self._frame_times <= stop))
            count = len(indices)
            if len(indices) > max_points:
                indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=int)]
            times = self._frame_times[indices]
            values = np.asarray([self._seed.array(self._seed.payload["native_inputs"][int(i)])[row] for i in indices])
            recording = "continuous"
        elif (pop, key) in self._window_fields and start <= self.continuous_window[1] and stop >= self.continuous_window[0]:
            meta = self._window_fields[pop, key]
            match = np.flatnonzero(meta['rows'] == row)
            if len(match):
                times = self._window_time
                values = self._window_values(pop, key)[match[0]]
            old_times = sorted({x[0] for x in self._snapshots.get((pop, key), [])
                if x[0] < self.continuous_window[0] or x[0] > self.continuous_window[1]})
            if old_times:
                old_values = [self._snapshot_values(pop, key, t)[row] for t in old_times]
                times = np.r_[times, old_times]; values = np.r_[values, old_values]
                order = np.argsort(times); times, values = times[order], values[order]
            recording = 'windowed'
        else:
            times = np.asarray(sorted({x[0] for x in self._snapshots.get((pop, key), [])}))
            values = np.asarray([self._snapshot_values(pop, key, t)[row] for t in times])
        if count is None:
            count = int(np.count_nonzero((times >= start) & (times <= stop)))
        preserve_extrema = (pop, row) == self.target and key in ("V", "gE", "gI")
        times, values = self._subsample(times, values, start, stop, max(2, int(max_points)), preserve_extrema)
        sampling = "原始点" if count == len(times) else ("保留分段极值抽样" if preserve_extrema else "原时标抽样")
        return dict(time=times, values=values, unit=info["unit"], label=info["label"],
            recording=recording, source_count=count, displayed_count=len(times),
            continuous_window=self.continuous_window if recording == 'windowed' else None,
            sampling=sampling,
            status=(f"已保存 · {sampling} {len(times)}/{count}" if np.isfinite(values).any() else "此对象/时间段未记录"))

    def _frame(self, seconds):
        if not np.isfinite(seconds) or not 0 <= seconds <= self.duration:
            raise ValueError("Time outside saved 0–45 s interval")
        return int(np.clip(np.searchsorted(self._frame_times, seconds, side="right") - 1, 0, len(self._frame_times) - 1))

    def spatial(self, pop, key, seconds):
        info = self.variables(pop).get(key, dict(unit="", label=key))
        actual = float(seconds)
        recording = "snapshot"
        if pop == "cones" and key == "native_input":
            frame = self._frame(seconds)
            values = np.asarray(self._seed.array(self._seed.payload["native_inputs"][frame]))
            actual, recording = float(self._frame_times[frame]), "continuous"
        elif key == "spike":
            actual = min(np.floor(seconds * 1000) / 1000, self.duration - .001)
            values = np.zeros(len(self.geometry[pop]["ids"]))
            for event in self._events.get(pop, []):
                if actual <= event["time_seconds"] < actual + .001:
                    values[event["cell_index"]] += 1
            recording = "saved_events_in_1ms_bin"
        else:
            values = self._snapshot_values(pop, key, seconds)
            windowed = ((pop, key) in self._window_fields
                and self.continuous_window[0] <= seconds <= self.continuous_window[1])
            if windowed:
                index = max(0, int(np.searchsorted(self._window_time, seconds, side='right') - 1))
                values[:] = np.nan
                values[self._window_fields[pop, key]['rows']] = self._window_values(pop, key)[:, index]
                actual = float(self._window_time[index])
                recording = 'replayed_window'
            if pop == self.target[0] and key in ("V", "gE", "gI", "expected_rate_Hz", "glm_rate_Hz"):
                dt = .0001 if key in ("V", "gE", "gI") else .001
                query = min(int(np.floor((actual if windowed else seconds) / dt + 1e-9)), int(self.duration / dt) - (dt == .001)) * dt
                result = self.series(pop, self.target[1], key, max(0, query - dt / 2), query + dt / 2, 2)
                if len(result["time"]):
                    values[self.target[1]] = result["values"][0]
                    actual = float(result["time"][0])
                recording = "continuous_target_and_exact_snapshots"
        return dict(values=values, time=actual, unit=info["unit"], label=info["label"],
            recording=recording, status="已保存" if np.isfinite(values).any() else "所选时刻未记录；请选择保存快照")

    def comparison(self, start=0., stop=45., max_points=2000):
        times = self._npz("readout/FIXED_DEVELOPMENT_PREDICTIONS.npz", "time_bin_start_seconds")[::20]
        measured = self._npz("readout/FIXED_DEVELOPMENT_PREDICTIONS.npz", "observed_counts").mean(axis=0).reshape(-1, 20).mean(axis=1) * 1000
        glm = self._npz("readout/FIXED_DEVELOPMENT_PREDICTIONS.npz", "probability").reshape(-1, 20).mean(axis=1) * 1000
        base = self._npz("PREFIX_45s.npz", "expected_rate_base_Hz").reshape(-1, 20).mean(axis=1)
        result = dict(unit="Hz", bin_width_s=.02, label="69#6 · 六次实测 / 冻结 GLM / 原 RGC", status="已保存预测；20 ms PSTH")
        for key, values in (("measured", measured), ("glm", glm), ("base", base)):
            result["time"], result[key] = self._subsample(times, values, start, stop, max_points)
        return result

    def _optical_source(self, seconds):
        frame = self._frame(seconds)
        info = self._optics["frames"][frame]
        if "short_capture_row" in info:
            folder = self.report / "optical"
            row = int(info["short_capture_row"])
        else:
            folder = self.report / "formal_optics" / f"job_{info['job']:04d}"
            row = int(info["capture_row"])
        return info, folder, row

    def _mat(self, path, key):
        return self._remember((str(path), key), lambda: loadmat(path, variable_names=[key])[key])

    def optical_capture(self, seconds):
        info, folder, row = self._optical_source(seconds)
        if self._optical_geometry is None:
            with np.load(self.report / "optical/APERTURE_GEOMETRY.npz", allow_pickle=False) as z:
                self._optical_geometry = dict(xy_mm=z["positions_mm"], ids=z["cone_ids"], subtypes=z["cone_types"])
        values = self._mat(folder / "CAPTURE.mat", "expected_counts")[row]
        return dict(**self._optical_geometry, values=values, unit="expected R*/frame", time=info["time_frame_start_seconds"],
            exposure_seconds=1 / 150, status="已保存 ISETBio 捕获；冻结 Gaussian 孔径；借用的人 Cone 晶格")

    def stimulus(self, seconds):
        info, folder, row = self._optical_source(seconds)
        bank = folder / "SOURCE_BANK.mat"
        if not bank.exists():
            declared = Path(_read(folder / "CONFIG.json")["source_bank"])
            bank = declared if declared.exists() else bank
        if not bank.exists():
            return dict(image=None, time=info["time_frame_start_seconds"], status="原刺激帧文件不可用")
        image = self._mat(bank, "source_rgb")[:, :, :, row]
        return dict(image=image, time=info["time_frame_start_seconds"], source_frame=info["source_frame_zero_based"], status="保存的原始 RGB 刺激帧")

    def close(self):
        self._cache.clear()
        self._cache_bytes = 0
