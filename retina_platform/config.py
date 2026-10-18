"""Explicit engineering media profile; never a physiological calibration switch."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import math

AUTHOR_STEP = .0001

@dataclass
class RunConfig:
    media_path: str = ''
    start_seconds: float = 0.
    duration_seconds: float = .006
    warmup_seconds: float = .002
    window_seconds: float = .002
    sample_seconds: float = .001
    effective_cones: int = 64
    effective_rods: int = 96
    field_width_deg: float = 70.
    native_background_td: float = 100.
    cone_background_rstar_s: float = 1000.
    rod_background_rstar_s: float = 5.
    gaze_x_deg: float = 0.
    gaze_y_deg: float = 0.
    gaze_csv: str = ''
    seed: int = 20261010
    execution_profile: str = 'retina_contact_geometry_effective_v1'
    circuit_configuration_path: str = ''
    parameter_profile: str = 'effective_dynamics_v1'
    adaptation_policy: str = 'uniform_declared_background'
    background_source: str = 'RunConfig_declared_background'
    rgb_adapter: str = 'iset_lcd_apple_mw_v1'
    frontend_signature: str = ''
    native_background_values: tuple | None = None
    reference_linear_value: float = .5
    owner_x_mm: float = .35
    owner_y_mm: float = 0.
    owner_width_mm: float = .02
    owner_height_mm: float = .02
    halo_mm: float = .04
    max_spatial_cells: int = 60000
    roi_preset: str = 'custom'
    hc_geometry_profile: str = 'hc_joint_synthetic_v1'
    hc_arbor_profile: str = 'W2000_equal_area_bounded_edges_047_104'
    ipl_geometry_profile: str = 'retina_contact_geometry_effective_v1'
    requested_owner_output: str = 'midget_RGC_ON'
    hc_geometry_path: str = 'configurations/hc_research_prototype_geometry_v1.json'
    hc_h1_radius_mm: float | None = None
    hc_h2_radius_mm: float | None = None
    background_initialization_mode: str = 'fixed_prerun'
    background_max_sim_seconds: float = 10.
    background_max_wall_seconds: float = 60.
    background_require_converged: bool = False
    background_policy_path: str = ''
    background_checkpoint_path: str = ''
    background_memory_limit_mb: float = 4096.
    background_min_available_mb: float = 1024.
    background_checkpoint_interval_seconds: float = 45.
    neural_integration_backend: str = 'scipy_cpu_v1'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def project_default(cls,root):
        import json
        policy=json.loads((Path(root)/'configurations/retina_platform_v1.json').read_bytes())
        if policy.get('schema_version')!='1.0':raise ValueError('Unknown shared platform configuration schema')
        return cls.from_dict(policy['run_defaults'])

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('配置必须是 JSON 对象')
        names = {f.name for f in fields(cls)}
        unknown = set(value) - names
        if unknown:
            raise ValueError('未知配置项: ' + ', '.join(sorted(unknown)))
        value=dict(value)
        if value.get('execution_profile')=='atlas_isetbio_geometry_v1':
            import json
            defaults=json.loads((Path(__file__).resolve().parents[1]/'configurations/retina_platform_v1.json').read_bytes())['run_defaults']
            for key,default in defaults.items():
                if key.startswith('background_'):value.setdefault(key,'' if key=='background_checkpoint_path' else default)
        if value.get('execution_profile') in ('atlas_midget_research_v1','atlas_population_revision_v1','atlas_isetbio_geometry_v1'):
            for key,default in {'warmup_seconds':0.,'duration_seconds':.0012,'window_seconds':.0012,
                'sample_seconds':.0001,'owner_x_mm':.35,'max_spatial_cells':60000,
                'hc_geometry_profile':'atlas_fixed_joint_master_v2','hc_arbor_profile':'atlas_source_arbor_v1',
                'hc_geometry_path':'configurations/atlas_hc_joint_master_v2.json',
                'ipl_geometry_profile':'atlas_typed_inner_v1'}.items():value.setdefault(key,default)
        elif value.get('execution_profile')!='retina_contact_geometry_effective_v1':
            for key,default in {'ipl_geometry_profile':'legacy_point_ipl_v1','hc_geometry_profile':'source_patch_v1',
                'hc_arbor_profile':'explicit_footprints','hc_geometry_path':'','owner_x_mm':4.6,'max_spatial_cells':8192}.items():
                value.setdefault(key,default)
        else:
            for key,default in {'ipl_geometry_profile':'retina_contact_geometry_effective_v1',
                'hc_geometry_profile':'hc_joint_synthetic_v1','hc_arbor_profile':'W2000_equal_area_bounded_edges_047_104',
                'hc_geometry_path':'configurations/hc_research_prototype_geometry_v1.json',
                'owner_x_mm':.35,'max_spatial_cells':60000,'roi_preset':'custom'}.items():value.setdefault(key,default)
        value.setdefault('roi_preset','custom')
        if 'execution_profile' not in value:
            value['execution_profile']='engineering_preview'
            value.setdefault('adaptation_policy','legacy_first_frame')
            value.setdefault('rgb_adapter','legacy_rgb_proxy_v1')
            value.setdefault('parameter_profile','legacy_effective_v1')
        return cls(**value)

    def validate(self, media=None):
        if not isinstance(self.media_path, str) or not Path(self.media_path).is_file():
            raise ValueError('请选择可读取的图像或视频')
        numeric = [f.name for f in fields(self) if f.name not in ('media_path', 'gaze_csv', 'execution_profile',
                   'parameter_profile', 'adaptation_policy', 'background_source','rgb_adapter','roi_preset',
                   'frontend_signature','native_background_values','circuit_configuration_path',
                   'hc_geometry_profile','hc_arbor_profile','hc_geometry_path','hc_h1_radius_mm','hc_h2_radius_mm',
                   'ipl_geometry_profile','requested_owner_output',
                   'neural_integration_backend',
                   'background_initialization_mode','background_policy_path','background_checkpoint_path','background_require_converged')]
        from retina_sim.execution_backends import BACKENDS
        if self.neural_integration_backend not in BACKENDS:
            raise ValueError('Unknown explicit numerical integration backend')
        if self.background_initialization_mode not in ('fixed_prerun','bounded_convergence'):
            raise ValueError('Unknown background initialization mode')
        if not isinstance(self.background_require_converged,bool):raise ValueError('background_require_converged must be boolean')
        if not isinstance(self.circuit_configuration_path,str):raise ValueError('Circuit configuration path must be text')
        if self.circuit_configuration_path and self.execution_profile!='atlas_isetbio_geometry_v1':
            raise ValueError('Explicit circuit configuration requires the frozen ISET geometry route')
        if self.background_initialization_mode=='bounded_convergence':
            if self.adaptation_policy!='uniform_declared_background':raise ValueError('Bounded preparation requires declared uniform background')
            self.background_policy()
        if self.hc_geometry_profile not in ('source_patch_v1','hc_joint_synthetic_v1','atlas_fixed_joint_master_v2'):
            raise ValueError('Unknown HC geometry profile')
        if not isinstance(self.hc_geometry_path,str):raise ValueError('HC geometry JSON path must be text')
        for name in ('hc_h1_radius_mm','hc_h2_radius_mm'):
            value=getattr(self,name)
            if value is not None and (isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0):
                raise ValueError(name+' requires a positive finite mm radius or an empty unconfigured value')
        if self.roi_preset not in ('custom','center_design','central_author_anchor','central_OPL_support','transition_anchor','peripheral_anchor'):
            raise ValueError('Unknown named ROI preset')
        if self.execution_profile not in ('engineering_preview', 'source_constrained_approximation','retina_contact_geometry_effective_v1','atlas_midget_research_v1','atlas_population_revision_v1','atlas_isetbio_geometry_v1'):
            raise ValueError('未知运行模式；全生理严格模式尚未标定')
        from retina_sim.ipl_geometry import read_ipl_geometry,TYPED_PROFILES
        if self.execution_profile not in ('atlas_midget_research_v1','atlas_population_revision_v1','atlas_isetbio_geometry_v1'):
            read_ipl_geometry(Path(__file__).resolve().parents[1],self.ipl_geometry_profile)
            if self.hc_geometry_profile=='atlas_fixed_joint_master_v2':raise ValueError('Fixed atlas HC belongs to the atlas Run profile')
        if self.ipl_geometry_profile in TYPED_PROFILES and self.execution_profile=='engineering_preview':
            raise ValueError('P-chart IPL需要源约束ROI；旧工程preview请选legacy IPL')
        if self.requested_owner_output not in ('midget_RGC_ON','midget_RGC_OFF','parasol_RGC_ON','parasol_RGC_OFF','small_bistratified_S_ON'):
            raise ValueError('Unknown requested owner RGC output')
        if self.parameter_profile not in ('legacy_effective_v1','effective_dynamics_v1'):
            raise ValueError('未知实际参数 profile')
        from retina_sim.hc_geometry import read_hc_geometry
        if self.execution_profile in ('atlas_midget_research_v1','atlas_population_revision_v1','atlas_isetbio_geometry_v1'):
            if self.execution_profile=='atlas_isetbio_geometry_v1' and (self.background_initialization_mode!='bounded_convergence' or not self.background_require_converged):
                raise ValueError('新ISET几何需要独立的有界收敛背景；旧几何checkpoint不能复用')
            if (self.hc_geometry_profile,self.hc_arbor_profile,self.ipl_geometry_profile)!=('atlas_fixed_joint_master_v2','atlas_source_arbor_v1','atlas_typed_inner_v1'):
                raise ValueError('Atlas Run requires its fixed actual geometry; old geometry settings are not used')
            if self.parameter_profile!='effective_dynamics_v1' or self.adaptation_policy!='uniform_declared_background':
                raise ValueError('Atlas Run requires the named effective dynamics and declared uniform background')
            if self.warmup_seconds!=0 or (self.execution_profile=='atlas_midget_research_v1' and self.background_initialization_mode!='fixed_prerun'):
                raise ValueError('Atlas requires zero fixed prerun; complete population graph also supports bounded background preparation')
            if self.hc_h1_radius_mm is not None or self.hc_h2_radius_mm is not None:
                raise ValueError('Atlas Run uses frozen source arbor bounds; radius overrides are not supported')
            if self.execution_profile=='atlas_midget_research_v1' and self.requested_owner_output not in ('midget_RGC_ON','midget_RGC_OFF'):
                raise ValueError('Atlas Run scope is owner midget ON/OFF; other paths remain explicitly unbound')
            hc=None
        else:
            hc=read_hc_geometry(Path(__file__).resolve().parents[1],self.hc_geometry_profile,
            config_path=self.hc_geometry_path,h1_radius_mm=self.hc_h1_radius_mm,h2_radius_mm=self.hc_h2_radius_mm,
            arbor_profile_id=self.hc_arbor_profile)
        if hc is not None and self.hc_geometry_profile=='hc_joint_synthetic_v1':
            if self.execution_profile=='engineering_preview':
                raise ValueError('联合HC合成需要源约束混合模式')
            from retina_sim.hc_arbor import validate_footprints
            validate_footprints(hc['footprints'],require_configured=True)
        self.light_context()
        if self.rgb_adapter!='iset_lcd_apple_mw_v1':
            raise ValueError('旧 RGB 输入仅供历史结果读取；运行需要当前 LCD-Apple/MW 混合参考前端')
        if self.native_background_td!=100 or self.reference_linear_value!=.5:
            raise ValueError('已批准参考固定为 100 td 与既有 reference u=0.5；更改需新参考配置')
        for name in numeric:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(name + ' 必须是有限数值')
        for name in ('effective_cones', 'effective_rods', 'seed','max_spatial_cells'):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(name + ' 必须是整数')
        if self.seed < 0 or (self.execution_profile=='engineering_preview' and (self.effective_cones < 64 or self.effective_rods < 64)):
            raise ValueError('随机种子非负；preview 的显式有效 cone 和 rod 数量至少 64')
        for name in ('duration_seconds', 'window_seconds', 'sample_seconds', 'field_width_deg',
                     'native_background_td', 'cone_background_rstar_s', 'rod_background_rstar_s',
                     'owner_width_mm','owner_height_mm','halo_mm','max_spatial_cells',
                      'background_max_sim_seconds','background_max_wall_seconds','background_memory_limit_mb','background_min_available_mb','background_checkpoint_interval_seconds'):
            if getattr(self, name) <= 0:
                raise ValueError(name + ' 必须大于 0')
        if self.start_seconds < 0 or self.warmup_seconds < 0:
            raise ValueError('开始时间与适应时间不能为负')
        for name in ('duration_seconds', 'warmup_seconds', 'window_seconds', 'sample_seconds','background_max_sim_seconds'):
            ticks = getattr(self, name) / AUTHOR_STEP
            if abs(ticks - round(ticks)) > 1e-7:
                raise ValueError(name + ' 须是 0.1 ms 的整数倍（作者离散时间步）')
        if self.sample_seconds > self.window_seconds:
            raise ValueError('采样间隔不能超过窗口长度')
        if not isinstance(self.gaze_csv, str):
            raise ValueError('眼位轨迹必须是 CSV 路径')
        if self.gaze_csv and not Path(self.gaze_csv).is_file():
            raise ValueError('未找到眼位轨迹 CSV')
        if media:
            info = media.info if hasattr(media, 'info') else media
            if info['kind'] == 'video' and self.start_seconds + self.duration_seconds > info['duration_seconds'] + 1e-9:
                raise ValueError('所选时间段超过视频长度')
        return self

    def background_policy(self):
        import json
        from retina_sim.background import BackgroundPolicy
        path=Path(self.background_policy_path) if self.background_policy_path else Path(__file__).resolve().parents[1]/'configurations/bounded_uniform_background_v1.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version')!='1.0' or not isinstance(data.get('policy'),dict):raise ValueError('Invalid background numerical policy JSON')
        parameters=dict(data['policy'],max_sim_seconds=self.background_max_sim_seconds,
            max_wall_seconds=self.background_max_wall_seconds,require_converged=self.background_require_converged)
        return BackgroundPolicy(**parameters)

    def light_context(self):
        from retina_sim.runtime_profiles import LightContext
        if self.rgb_adapter=='iset_lcd_apple_mw_v1':
            return LightContext(self.native_background_td,self.cone_background_rstar_s,self.rod_background_rstar_s,
                'iset_lcd_apple_mw_reference_v1','ISET_LCD_Apple_human_MW_mixed_reference',
                'approved_mixed_reference:'+self.frontend_signature,self.adaptation_policy,'native_td',
                self.rgb_adapter,self.reference_linear_value,self.field_width_deg,self.native_background_values)
        return LightContext(self.native_background_td, self.cone_background_rstar_s,
            self.rod_background_rstar_s, self.background_source, 'RGB_display_approximation',
            'uncalibrated_sRGB_weighted_display_mapping', self.adaptation_policy,'relative_intensity',
            self.rgb_adapter,self.reference_linear_value,self.field_width_deg)

    def owner_bounds(self):
        return (self.owner_x_mm-self.owner_width_mm/2,self.owner_x_mm+self.owner_width_mm/2,
            self.owner_y_mm-self.owner_height_mm/2,self.owner_y_mm+self.owner_height_mm/2)

def input_assumptions(config):
    if config.rgb_adapter=='iset_lcd_apple_mw_v1':
        return {'profile':'iset_lcd_apple_mw_v1','scope':'approved_mixed_reference',
            'input_class':'LCD-Apple gamma/SPD -> scaled human-MW -> ISET cone capture -> fixed LMS reference-equivalent native drive',
            'frontend_signature':config.frontend_signature,'evidence_class':'project_approximation',
            'physiological_calibration':'MIXED_REFERENCE_NOT_FULL_MACAQUE','spectral_physical_calibration':False,
            'rgb_transfer':'digital RGB drives measured LCD gamma once; no preceding sRGB inverse for cones',
            'rod_input':'existing independent sRGB weighted rod route, not ISET capture',
            'weights':{'rod':[.2126,.7152,.0722]},'reference_linear_value':config.reference_linear_value,
            'field_width_deg':config.field_width_deg,'retinal_mm_per_degree':.223,
            'light_context':config.light_context().manifest(config.warmup_seconds),
            'includes_background':True,'native_interpolation':'held 100us integration-window means'}
    return {
        'profile': 'engineering_sRGB_profile', 'input_class': 'RGB_media_with_synthetic_engineering_light_mapping',
        'evidence_class': 'project_approximation', 'spectral_physical_calibration': False,
        'synthetic': True, 'rgb_transfer': 'sRGB inverse transfer; input files assumed sRGB',
        'weights': {'L': [.6, .3, .1], 'M': [.2, .7, .1], 'S': [.1, .1, .8], 'rod': [.2126, .7152, .0722]},
        'adapter_id':config.rgb_adapter,
        'reference_linear_value':config.reference_linear_value,
        'rate_mapping': 'background * weighted_linear_RGB / reference_linear_value' if config.rgb_adapter=='contrast_reference_v1' else 'background * (0.6 + 0.8 * weighted_linear_RGB)',
        'backgrounds': {'native': {'value': config.native_background_td, 'unit': 'td'},
                        'physical_cone': {'value': config.cone_background_rstar_s, 'unit': 'R*/s'},
                        'rod': {'value': config.rod_background_rstar_s, 'unit': 'R*/s'}},
        'field_width_deg': config.field_width_deg, 'retinal_mm_per_degree': .223,
        'retinal_magnification_class': 'project_approximation; existing full_target geometry conversion',
        'image_xy_sign': 'x right, y down; retinal RF offset + prescribed gaze',
        'outside_image': 'uniform reference_linear_value RGB' if config.rgb_adapter=='contrast_reference_v1' else 'uniform 0.5 linear RGB', 'includes_background': True,
        'native_interpolation': 'linear at 0.0001 s grid', 'physical_interpolation': 'zero_order_hold at author 0.0001 s',
        'light_context': config.light_context().manifest(config.warmup_seconds),
        'warmup': config.background_initialization_mode + '; actual status exported separately; absolute state and seeded RNG retained',
        'background_initialization': {'mode':config.background_initialization_mode,
            'policy_path':config.background_policy_path,'max_sim_seconds':config.background_max_sim_seconds,
            'max_wall_seconds':config.background_max_wall_seconds,'require_converged':config.background_require_converged},
        'gaze': 'prescribed CSV with time_seconds,x_deg,y_deg, linear; stimulus time starts at 0' if config.gaze_csv else 'fixed prescribed gaze',
        'geometry_request': ({'basis':'density_integral','owner_bounds_mm':list(config.owner_bounds()),
            'halo_mm':config.halo_mm,'max_population_cells':config.max_spatial_cells,'roi_preset':config.roi_preset,
            'IPL_geometry_profile':config.ipl_geometry_profile,'requested_owner_output':config.requested_owner_output,
            'HC_geometry':{'profile_id':config.hc_geometry_profile,'configuration_path':config.hc_geometry_path,
                'arbor_profile_id':config.hc_arbor_profile,
                'H1_explicit_radius_mm':config.hc_h1_radius_mm,'H2_explicit_radius_mm':config.hc_h2_radius_mm,
                'no_implicit_central_arbor_radius':True,'implementation_status':'IMPLEMENTED_NOT_RUN'},
            'region_labels':'project navigation only; source recording frames and parameter domains remain independent'}
            if config.execution_profile!='engineering_preview' else
            {'basis':'effective_sampling','cones':config.effective_cones,'rods':config.effective_rods,
             'bounds_mm':[.004,7.2,-.04,.04]}),
        'physiological_calibration': 'NOT_COMPLETE', 'independent_animal_validation': 'NOT_RUN',
    }
