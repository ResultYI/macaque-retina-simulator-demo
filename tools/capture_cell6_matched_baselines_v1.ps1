param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$here='C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision'
if(Test-Path -LiteralPath $OutputDirectory){throw 'Existing native capture retained; no automatic retry'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$watch=[Diagnostics.Stopwatch]::StartNew();$nonce=[Guid]::NewGuid().ToString('N')
$worker=$null;$status='FAILED';$Mode='stimulus_LN_GLM_matched_input_supplement';$Allocator='debug';$scriptName='run_cell6_matched_baselines_v1.py'
function Record($name,$value){[IO.File]::WriteAllText((Join-Path $OutputDirectory $name),($value|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))}
function OwnedWorker {
    if($null -eq $worker){return $null}
    $observed=Get-CimInstance Win32_Process -Filter "ProcessId=$($worker.identity.pid)"
    if($null -eq $observed){return $null}
    if($observed.CommandLine -notlike "*$nonce*" -or $observed.CommandLine -notlike "*$scriptName*"){throw 'Owned debugger worker identity mismatch'}
    if((Get-Process -Id $worker.identity.pid).StartTime.ToFileTimeUtc().ToString() -ne $worker.identity.creation_100ns){throw 'Owned debugger worker creation mismatch'}
    return $observed
}
try {
    Add-Type -Path @((Join-Path $here 'owned_native_debug_v4.cs'),(Join-Path $here 'python_frame_capture_v4.cs'))
    $arguments=@('-B',('"'+'D:\macaque retinal simulator\tools\run_cell6_matched_baselines_v1.py'+'"'),$nonce,('"'+$OutputDirectory+'"'))

    $launcher=Start-Process -FilePath 'D:\macaque retinal simulator\.venv\Scripts\python.exe' -ArgumentList $arguments -Environment @{PYTHONMALLOC=$Allocator} -WorkingDirectory 'D:\macaque retinal simulator' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputDirectory 'STDOUT.log') -RedirectStandardError (Join-Path $OutputDirectory 'STDERR.log')
    $gate=[DateTime]::UtcNow.AddSeconds(30)
    while(-not(Test-Path -LiteralPath (Join-Path $OutputDirectory 'WORKER_READY.json'))){
        if($launcher.HasExited -or [DateTime]::UtcNow -gt $gate){throw 'Own worker did not reach attachment gate'}
        Start-Sleep -Milliseconds 100
    }
    $worker=Get-Content -LiteralPath (Join-Path $OutputDirectory 'WORKER_READY.json') -Raw|ConvertFrom-Json
    if($worker.nonce -ne $nonce -or $null -eq (OwnedWorker)){throw 'Own worker gate failed'}
    Record 'CONTROLLER.json' @{pid=$PID;worker_pid=$worker.identity.pid;launcher_pid=$launcher.Id;mode=$Mode;allocator=$Allocator;direct_native_API=$true;UI_used=$false;utc=[DateTime]::UtcNow.ToString('o')}
    $exit=[OwnedNativeDebugV4]::Run([uint32]$worker.identity.pid,$OutputDirectory,$nonce)
    $status=if(Test-Path -LiteralPath (Join-Path $OutputDirectory 'EXACT_NATIVE_EXCEPTION.json')){'NATIVE_FAILURE_CAPTURED'}elseif($exit -eq 0){'EXITED_0'}else{'WORKER_FAILED'}
} catch {
    $inner=$_.Exception;while($null -ne $inner.InnerException){$inner=$inner.InnerException}
    Record 'CONTROLLER_ERROR.json' @{error=$_.Exception.Message;inner=$inner.Message;native_error=if($inner -is [ComponentModel.Win32Exception]){$inner.NativeErrorCode}else{$null};stack=$_.ScriptStackTrace}
} finally {
    if($null -ne $worker){try{if($null -ne (OwnedWorker)){Stop-Process -Id $worker.identity.pid;Record 'OWN_WORKER_STOPPED.json' @{reason='Controller fail-stop cleanup';pid=$worker.identity.pid}}}catch{Record 'CLEANUP_ERROR.json' @{error=$_.Exception.Message}}}
    Record 'TERMINAL.json' @{status=$status;wall_seconds=$watch.Elapsed.TotalSeconds;automatic_retry=$false;no_system_or_model_changes=$true}
}

if($status -eq 'EXITED_0' -and (Test-Path -LiteralPath (Join-Path $OutputDirectory 'COMPUTATION_COMPLETE.json'))){
    $plotArgs=@('-B','"D:\macaque retinal simulator\tools\finish_cell6_matched_baselines_v1.py"',('"'+$OutputDirectory+'"'))
    $plot=Start-Process -FilePath 'D:\anaconda\python.exe' -ArgumentList $plotArgs -WorkingDirectory 'D:\macaque retinal simulator' -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $OutputDirectory 'DELIVERY_STDOUT.log') -RedirectStandardError (Join-Path $OutputDirectory 'DELIVERY_STDERR.log')
    Record 'PLOT_DELIVERY_TERMINAL.json' @{exit_code=$plot.ExitCode;pid=$plot.Id;new_fits=0;new_predictions=0;new_evaluations=0;utc=[DateTime]::UtcNow.ToString('o')}
}
