"""Per-round user override of application walltime only; never resource/QA gates."""
from pathlib import Path
import hashlib,json
from .contracts import ContractError

OVERRIDE='user_removed_round_walltime.json'
EXTENDED_GOAL='overnight_goal_authorization.json'
OVERNIGHT_SCOPE='OVERNIGHT_EXISTING_TWO_SHARED_GAIN_DISTILLATION'

def extended_goal(directory,budget):
    path=Path(directory)/EXTENDED_GOAL
    if not path.exists():return None
    value=json.loads(path.read_bytes())
    if (value.get('schema')!='explicit-overnight-existing-fit-goal-1'
            or value.get('scope')!=OVERNIGHT_SCOPE
            or value.get('original_budget_sha256')!=hashlib.sha256((Path(directory)/'round_budget.json').read_bytes()).hexdigest()
            or value.get('input_policy_sha256')!=budget['policy_sha256']
            or not value.get('trusted_user_instruction_reference')
            or value.get('max_rss_bytes')!=budget['max_rss_bytes']
            or value.get('max_total_gpu_bytes')!=budget['max_total_gpu_bytes']
            or value.get('parameter_freedom')!='original_two_shared_EI_gains_0.5_to_2'
            or value.get('original_750_background_frames_preserved') is not True
            or value.get('splits_seconds')!={'train':[0,35],'validation':[40,50],'preliminary_test':[55,60],'future_final_untouched':[480,600]}):
        raise ContractError('Extended overnight goal changed the approved input/resources/fit scope')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def effective_deadline(directory,budget):
    directory=Path(directory);path=directory/OVERRIDE
    if not path.exists():return budget['deadline_unix']
    value=json.loads(path.read_bytes())
    if (value.get('schema')!='explicit-useful-round-walltime-removal-1'
            or value.get('original_budget_sha256')!=hashlib.sha256((directory/'round_budget.json').read_bytes()).hexdigest()
            or value.get('input_policy_sha256')!=budget['policy_sha256']
            or value.get('user_removed_time_deadline') is not True
            or value.get('scope')!='148_OPTICAL_TILES_QUALIFIED_BACKGROUND_ONE_100MS_PREFIX'
            or not value.get('trusted_user_instruction_reference')
            or value.get('max_rss_bytes')!=budget['max_rss_bytes']
            or value.get('max_total_gpu_bytes')!=budget['max_total_gpu_bytes']):
        raise ContractError('Runtime override changed scope/resources or lacks explicit user authorization')
    return None

def runtime_limits(directory,budget):
    """Authorization fields; original round file remains immutable history."""
    directory=Path(directory).resolve();deadline=effective_deadline(directory,budget)
    value=dict(round_started_unix=budget['started_unix'],expires_unix=deadline,
        max_wall_seconds=None if deadline is None else budget['max_wall_seconds'],
        max_rss_bytes=budget['max_rss_bytes'],max_vram_bytes=budget['max_total_gpu_bytes'])
    if deadline is None:
        value.update(time_limit_mode='USER_REMOVED_THIS_ROUND_WALLTIME',
            authorized_goal='148_OPTICAL_TILES_QUALIFIED_BACKGROUND_ONE_100MS_PREFIX',
            runtime_round_directory=str(directory),
            user_walltime_override_sha256=hashlib.sha256((directory/OVERRIDE).read_bytes()).hexdigest())
        extension=extended_goal(directory,budget)
        if extension is not None:
            value.update(authorized_goal=OVERNIGHT_SCOPE,extended_goal_authorization_sha256=extension)
    return value

def goal_only_authorization(authorization):
    if authorization.get('time_limit_mode')!='USER_REMOVED_THIS_ROUND_WALLTIME':return False
    directory=Path(authorization.get('runtime_round_directory','')).resolve()
    workspace=Path(__file__).resolve().parents[1]
    if not directory.is_relative_to(workspace):raise ContractError('Runtime goal outside current project')
    budget=json.loads((directory/'round_budget.json').read_bytes())
    if (effective_deadline(directory,budget) is not None
            or authorization.get('user_walltime_override_sha256')!=hashlib.sha256((directory/OVERRIDE).read_bytes()).hexdigest()
            or authorization.get('authorized_goal') not in ('148_OPTICAL_TILES_QUALIFIED_BACKGROUND_ONE_100MS_PREFIX',OVERNIGHT_SCOPE)
            or authorization.get('round_started_unix')!=budget['started_unix']
            or authorization.get('expires_unix') is not None or authorization.get('max_wall_seconds') is not None
            or authorization.get('max_rss_bytes')!=budget['max_rss_bytes']
            or authorization.get('max_vram_bytes')!=budget['max_total_gpu_bytes']):
        raise ContractError('User goal-only runtime identity/resources differ')
    if authorization.get('authorized_goal')==OVERNIGHT_SCOPE and (
            authorization.get('extended_goal_authorization_sha256')!=extended_goal(directory,budget)
            or authorization.get('extended_goal_authorization_sha256') is None):
        raise ContractError('Overnight source continuation has no matching explicit user goal')
    return True
