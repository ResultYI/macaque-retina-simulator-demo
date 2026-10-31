param([Parameter(Mandatory=$true)][string]$Experiment)
$ErrorActionPreference='Stop'
$root='D:\macaque retinal simulator'
$here='C:\Users\win11-pc\Documents\Codex\2026-10-29\task\hc_revision'
$pwsh='C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
$watch=[Diagnostics.Stopwatch]::StartNew();$stage='adopting_existing_cache';$status='RUNNING';$current=$null;$fit=$null
function ReadJson($path){
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete));$reader=[IO.StreamReader]::new($stream)
    try{return $reader.ReadToEnd()|ConvertFrom-Json}finally{$reader.Dispose()}
}
function WriteJson($path,$value){
    $tmp=$path+'.writing';[IO.File]::WriteAllText($tmp,($value|ConvertTo-Json -Depth 15),[Text.UTF8Encoding]::new($false));$deadline=[DateTime]::UtcNow.AddSeconds(2)
    while($true){
        try{[IO.File]::Move($tmp,$path,$true);return}
        catch{if([DateTime]::UtcNow -ge $deadline -or ($_.Exception.GetBaseException().HResult -band 65535) -notin @(5,32,33)){throw};Start-Sleep -Milliseconds 20}
    }
}
function NativeIdentity($number){
    $value=& (Join-Path $root '.venv\Scripts\python.exe') -B -c 'import json,sys;from tools.durable_research_runtime_v1 import process_identity;print(json.dumps(process_identity(int(sys.argv[1]))))' ([string]$number)
    if($LASTEXITCODE -ne 0){throw 'Cannot read exact controller identity'}
    return $value|ConvertFrom-Json
}
function State($extra){WriteJson (Join-Path $Experiment 'TRAIN_FIRST_STATE.json') @{pid=$PID;phase=$stage;status=$status;utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$watch.Elapsed.TotalSeconds;current_controller=$current;fit_controller=if($fit){$fit.Id}else{$null};extra=$extra;automatic_retry=$false}}
function VerifyFrozen {
    $freeze=ReadJson (Join-Path $Experiment 'TRAIN_FIRST_IMPLEMENTATION_FREEZE.json')
    foreach($property in $freeze.PSObject.Properties){
        if((Get-FileHash -LiteralPath (Join-Path $root $property.Name)).Hash.ToLowerInvariant() -ne $property.Value){throw ('Frozen implementation changed: '+$property.Name)}
    }
    if(Test-Path -LiteralPath (Join-Path $Experiment 'REQUEST_STOP')){throw 'Explicit STOP'}
}
function Alive($process){
    $observed=Get-Process -Id $process.Id -ErrorAction SilentlyContinue
    return $null -ne $observed -and $observed.StartTime.ToFileTimeUtc() -eq $process.StartTime.ToFileTimeUtc()
}
function FitHealthy {
    $terminalPath=Join-Path $Experiment ('capture\'+(CaptureName 'fit')+'\TERMINAL.json')
    if($fit -and (Test-Path -LiteralPath $terminalPath)){
        $terminal=ReadJson $terminalPath
        if($terminal.status -ne 'EXITED_0'){throw ('Readout stage failed: '+$terminal.status)}
    } elseif($fit -and -not(Alive $fit)){throw 'Readout controller disappeared without receipt'}
}
function CaptureName($name){
    $map=Join-Path $Experiment 'CACHE_CAPTURE_MAP.json'
    if(Test-Path -LiteralPath $map){$value=(ReadJson $map).$name;if($value){return $value}}
    return $name
}
function WaitStage($name,$process) {
    $directory=Join-Path $Experiment ('capture\'+(CaptureName $name));$terminal=Join-Path $directory 'TERMINAL.json'
    while(-not(Test-Path -LiteralPath $terminal)){
        VerifyFrozen;FitHealthy
        if(-not(Alive $process)){throw ('Controller disappeared without terminal: '+$name)}
        $path=Join-Path $Experiment ('cache\'+$name+'\PROGRESS.json')
        State @{progress=if(Test-Path -LiteralPath $path){ReadJson $path}else{$null}}
        Start-Sleep -Seconds 5
    }
    $result=ReadJson $terminal;$debug=ReadJson (Join-Path $directory 'DEBUG_TERMINAL.json')
    if($result.status -ne 'EXITED_0' -or -not $debug.worker_exited -or $debug.native_failure_captured){throw ('Failed stage retained: '+$name+' '+$result.status)}
    $resource=ReadJson (Join-Path $directory 'RESOURCE.json')
    if($resource.status -ne 'PASS'){throw ('Resource failure retained: '+$name)}
    if($name -ne 'fit'){
        $m=ReadJson (Join-Path $Experiment ('cache\'+$name+'\manifest.json'))
        if($m.status -ne 'COMPLETE_NEW_TOPOLOGY_CELL6_CACHE' -or $m.samples -ne 71001 -or $m.bins -ne 7100 -or $m.duration_seconds -ne 7.1 -or -not $m.physical_gate_passed -or -not $m.complete_research_restore -or $m.old_neural_cache_reused){throw ('Incomplete cache: '+$name)}
    }
    State @{completed_stage=$name;terminal=$result}
}
function StartCache($group){
    VerifyFrozen;$name='g'+$group;$output=Join-Path $Experiment ('capture\'+(CaptureName $name))
    if(Test-Path -LiteralPath $output){throw ('Existing stage retained: '+$name)}
    if($group -ge 1){
        $arguments=@('-B',('"'+(Join-Path $root 'tools\cell6_segmented_cache_controller_v1.py')+'"'),[string]$group,('"'+$output+'"'))
        return Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $arguments -WindowStyle Hidden -PassThru -WorkingDirectory $root -RedirectStandardOutput (Join-Path $Experiment ($name+'_CONTROLLER_STDOUT.log')) -RedirectStandardError (Join-Path $Experiment ($name+'_CONTROLLER_STDERR.log'))
    }
    $arguments=@('-NoProfile','-File',('"'+(Join-Path $here 'capture_cell6_monitor_repair_v1.ps1')+'"'),'-Group',[string]$group,'-OutputDirectory',('"'+$output+'"'))
    return Start-Process -FilePath $pwsh -ArgumentList $arguments -WindowStyle Hidden -PassThru -WorkingDirectory $root -RedirectStandardOutput (Join-Path $Experiment ($name+'_CONTROLLER_STDOUT.log')) -RedirectStandardError (Join-Path $Experiment ($name+'_CONTROLLER_STDERR.log'))
}
function CacheProcess($group){
    $directory=Join-Path $Experiment ('capture\'+(CaptureName ('g'+$group)))
    if(-not(Test-Path -LiteralPath $directory)){return StartCache $group}
    if(Test-Path -LiteralPath (Join-Path $directory 'TERMINAL.json')){return $null}
    $receipt=ReadJson (Join-Path $directory 'CONTROLLER.json');$identity=NativeIdentity $receipt.pid
    if($identity.state -ne 'ALIVE'){throw 'Existing cache controller unavailable without terminal; do not duplicate'}
    if($receipt.identity -and ($identity.creation_100ns -ne $receipt.identity.creation_100ns -or $identity.image -ine $receipt.identity.image)){throw 'Existing cache controller identity changed'}
    return Get-Process -Id $receipt.pid -ErrorAction Stop
}
$lock=Join-Path $Experiment 'TRAIN_FIRST_OWNER.json'
$stream=[IO.File]::Open($lock,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {
    $identity=NativeIdentity $PID
    if($identity.state -ne 'ALIVE'){throw 'Own continuation identity unavailable'}
    $owner=@{pid=$PID;creation_100ns=$identity.creation_100ns;image=$identity.image;started_utc=[DateTime]::UtcNow.ToString('o');automatic_retry=$false}
    $bytes=[Text.Encoding]::UTF8.GetBytes(($owner|ConvertTo-Json));$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)
} finally {$stream.Dispose()}
try {
    $takeover=ReadJson (Join-Path $Experiment 'TRAIN_FIRST_TAKEOVER.json')
    foreach($old in $takeover.stopped_controllers){
        $p=Get-Process -Id $old.pid -ErrorAction SilentlyContinue
        if($p -and $p.StartTime.ToFileTimeUtc().ToString() -eq $old.creation_100ns){throw 'Prior controller still owns scheduling'}
    }
    VerifyFrozen
    foreach($group in 0..2){
        $stage='g'+$group;$p=CacheProcess $group
        $current=if($p){$p.Id}else{$null};WaitStage $stage $p
    }
    if(-not (ReadJson (Join-Path $Experiment 'cache\g0\SHORT_NUMERICAL_CHECK.json')).passed){throw 'Original numerical check failed'}
    VerifyFrozen;$stage='training_g0_g2';$output=Join-Path $Experiment ('capture\'+(CaptureName 'fit'))
    if(Test-Path -LiteralPath $output){throw 'Existing fit retained; no second fit'}
    $fitScript=Join-Path $here 'capture_cell6_monitor_repair_fit_v1.ps1'
    if(Test-Path -LiteralPath (Join-Path $Experiment 'readout\FROZEN_IDENTITY.json')){
        $fitScript=Join-Path $root 'tools\capture_cell6_frozen_evaluation_resume_v1.ps1';$stage='resuming_frozen_fixed_evaluation'
    }
    $arguments=@('-NoProfile','-File',('"'+$fitScript+'"'),'-OutputDirectory',('"'+$output+'"'))
    $fitLog=(CaptureName 'fit').ToUpperInvariant()
    $fit=Start-Process -FilePath $pwsh -ArgumentList $arguments -WindowStyle Hidden -PassThru -WorkingDirectory $root -RedirectStandardOutput (Join-Path $Experiment ($fitLog+'_CONTROLLER_STDOUT.log')) -RedirectStandardError (Join-Path $Experiment ($fitLog+'_CONTROLLER_STDERR.log'))
    $current=$fit.Id;State @{started=$stage;training_restart=$false}
    while(-not(Test-Path -LiteralPath (Join-Path $Experiment 'readout\FROZEN_IDENTITY.json'))){
        VerifyFrozen;FitHealthy;State @{awaiting='frozen_training_parameters_and_standardization'};Start-Sleep -Seconds 5
    }
    foreach($group in 3..5){
        $stage='g'+$group;$p=CacheProcess $group;$current=if($p){$p.Id}else{$null};WaitStage $stage $p
    }
    $stage='fixed_evaluation_g3_g5';$current=$fit.Id;WaitStage 'fit' $fit
    $complete=ReadJson (Join-Path $Experiment 'readout\COMPLETE.json')
    if($complete.groups_evaluated -ne 6 -or $complete.temporal_fit_count -ne 1 -or $complete.historical_test_accessed){throw 'Fixed evaluation contract incomplete'}
    $status='COMPLETE';$stage='six_caches_one_fit_fixed_evaluations_complete';State @{readout=$complete}
} catch {
    $status='STOPPED_NO_RETRY'
    WriteJson (Join-Path $Experiment 'CONTINUATION_FAILURE.json') @{phase=$stage;error=$_.Exception.Message;stack=$_.ScriptStackTrace;utc=[DateTime]::UtcNow.ToString('o');automatic_retry=$false}
    State @{error=$_.Exception.Message}
} finally {
    WriteJson (Join-Path $Experiment 'CONTINUATION_TERMINAL.json') @{status=$status;phase=$stage;wall_seconds=$watch.Elapsed.TotalSeconds;automatic_retry=$false;new_scientific_experiment_added=$false;train_first=$true}
    if($status -ne 'COMPLETE' -and $fit -and (Alive $fit)){WriteJson (Join-Path $Experiment ('capture\'+(CaptureName 'fit')+'\STOP')) @{reason='Owned continuation failed; preserve frozen fit without retry'}}
}
$arguments=@('-B',('"'+(Join-Path $root 'tools\deliver_cell6_monitor_repair_v1.py')+'"'))
$delivery=Start-Process -FilePath 'D:\macaque retinal simulator\.venv\Scripts\python.exe' -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru -WorkingDirectory $root -RedirectStandardOutput (Join-Path $Experiment 'TRAIN_FIRST_DELIVERY_STDOUT.log') -RedirectStandardError (Join-Path $Experiment 'TRAIN_FIRST_DELIVERY_STDERR.log')
WriteJson (Join-Path $Experiment 'TRAIN_FIRST_DELIVERY_TERMINAL.json') @{exit_code=$delivery.ExitCode;reads_saved_results_only=$true;utc=[DateTime]::UtcNow.ToString('o')}
