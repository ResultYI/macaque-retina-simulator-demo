param([Parameter(Mandatory=$true)][ValidateSet('joint_prepare_native','joint_short','joint_segment','joint_train','joint_evaluate','install_owner_support','boundary_regression','build_master','build_bc','assemble_fine','extend_master','assemble_native','prepare_optical','assemble_groups','initialize','short_forward','prepare_formal_optical','prepare_native_prefix','formal_segment','train','evaluate')][string]$Stage,[Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$here='C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision';$root='D:\macaque retinal simulator'
if(Test-Path -LiteralPath $OutputDirectory){throw 'Existing native attempt retained'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$watch=[Diagnostics.Stopwatch]::StartNew();$nonce=[Guid]::NewGuid().ToString('N');$worker=$null;$status='FAILED';$scriptName='gin69_joint_worker.py'
function Record($name,$value){[IO.File]::WriteAllText((Join-Path $OutputDirectory $name),($value|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))}
function OwnedWorker {
    if($null -eq $worker){return $null}
    $observed=Get-CimInstance Win32_Process -Filter "ProcessId=$($worker.identity.pid)"
    if($null -eq $observed){return $null}
    if($observed.CommandLine -notlike "*$nonce*" -or $observed.CommandLine -notlike "*$scriptName*"){throw 'Owned worker command mismatch'}
    if((Get-Process -Id $worker.identity.pid).StartTime.ToFileTimeUtc().ToString() -ne $worker.identity.creation_100ns){throw 'Owned worker creation mismatch'}
    return $observed
}
try {
    Add-Type -Path @((Join-Path $here 'owned_native_debug_v4.cs'),(Join-Path $here 'python_frame_capture_v4.cs'))
    $arguments=@('-I','-S','-X','dev','-B',('"'+$root+'\tools\wp1_existing_python_runtime_v3.py"'),('"'+$root+'\tools\'+$scriptName+'"'),$nonce,('"'+$OutputDirectory+'"'),$Stage)
    $launcher=Start-Process -FilePath 'D:\anaconda\envs\snn_env\python.exe' -ArgumentList $arguments -Environment @{PYTHONMALLOC='debug';OMP_NUM_THREADS='1';OPENBLAS_NUM_THREADS='1';MKL_NUM_THREADS='1'} -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputDirectory 'STDOUT.log') -RedirectStandardError (Join-Path $OutputDirectory 'STDERR.log')
    $gate=[DateTime]::UtcNow.AddSeconds(30)
    while(-not(Test-Path -LiteralPath (Join-Path $OutputDirectory 'WORKER_READY.json'))){
        if($launcher.HasExited -or [DateTime]::UtcNow -gt $gate){throw 'Own worker did not reach attachment gate'}
        Start-Sleep -Milliseconds 100
    }
    $worker=Get-Content -LiteralPath (Join-Path $OutputDirectory 'WORKER_READY.json') -Raw|ConvertFrom-Json
    if($worker.nonce -ne $nonce -or $null -eq (OwnedWorker)){throw 'Own worker gate failed'}
    Record 'CONTROLLER.json' @{pid=$PID;worker_pid=$worker.identity.pid;stage=$Stage;utc=[DateTime]::UtcNow.ToString('o')}
    $exit=[OwnedNativeDebugV4]::Run([uint32]$worker.identity.pid,$OutputDirectory,$nonce)
    $status=if(Test-Path -LiteralPath (Join-Path $OutputDirectory 'EXACT_NATIVE_EXCEPTION.json')){'NATIVE_FAILURE_CAPTURED'}elseif($exit -eq 0){'EXITED_0'}else{'WORKER_FAILED'}
} catch {Record 'CONTROLLER_ERROR.json' @{error=$_.Exception.Message;stack=$_.ScriptStackTrace}}
finally {
    if($null -ne $worker){try{if($null -ne (OwnedWorker)){Stop-Process -Id $worker.identity.pid;Record 'OWN_WORKER_STOPPED.json' @{reason='Controller fail-stop cleanup';pid=$worker.identity.pid}}}catch{Record 'CLEANUP_ERROR.json' @{error=$_.Exception.Message}}}
    Record 'TERMINAL.json' @{status=$status;wall_seconds=$watch.Elapsed.TotalSeconds;automatic_retry=$false;native_root_cause_fixed_claimed=$false}
}
