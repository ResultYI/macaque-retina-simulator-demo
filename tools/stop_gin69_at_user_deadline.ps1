param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$base='D:\macaque retinal simulator\reports\GIN_69_6_HC2_BC1_SUPPORT_V2'
$pipeline=Join-Path $base 'pipeline_01'
$optical=Join-Path $base 'executions\formal_optics_45_01'
$deadline=[DateTimeOffset]::Parse('2026-11-02T00:00:00Z')
if(Test-Path -LiteralPath $OutputDirectory){throw 'Existing deadline receipt retained'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
function Record($name,$value){
    $path=Join-Path $OutputDirectory $name
    [IO.File]::WriteAllText(($path+'.tmp'),($value|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath ($path+'.tmp') -Destination $path -Force
}
function SameProcess($item){
    $p=Get-Process -Id $item.pid -ErrorAction SilentlyContinue
    if($null -eq $p){return $null}
    if($p.StartTime.ToUniversalTime().Ticks -ne $item.start_ticks -or $p.Path -ne $item.path){return $null}
    return $p
}
$specs=@(
    @{pid=21352;role='MATLAB computation';path='C:\Program Files\MATLAB\R2025b\bin\win64\MATLAB.exe';contains='capture_gin69_formal_optics_v1';parent=32200},
    @{pid=32200;role='MATLAB launcher';path='C:\Program Files\MATLAB\R2025b\bin\matlab.exe';contains='capture_gin69_formal_optics_v1';parent=36580},
    @{pid=36580;role='optical resource guard';path='C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe';contains='capture_gin69_formal_optics_support_v2.ps1';parent=39264},
    @{pid=39264;role='pipeline controller';path='C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe';contains='continue_gin69_support_v2.ps1';parent=27480}
)
$owned=@()
foreach($s in $specs){
    $p=Get-Process -Id $s.pid -ErrorAction Stop
    $c=Get-CimInstance Win32_Process -Filter ('ProcessId='+$s.pid)
    if($p.Path -ne $s.path -or $c.ParentProcessId -ne $s.parent -or -not $c.CommandLine.Contains($s.contains) -or -not $c.CommandLine.Replace('/','\').Contains($base)){throw ('Task identity mismatch: '+$s.pid)}
    $owned+=@{pid=$p.Id;role=$s.role;path=$p.Path;start_ticks=$p.StartTime.ToUniversalTime().Ticks;creation_utc=$p.StartTime.ToUniversalTime().ToString('o');parent=$c.ParentProcessId;command=$c.CommandLine}
}
# This existing gate prevents any next stage; the active optical call continues.
[IO.File]::WriteAllText((Join-Path $pipeline 'REQUEST_STOP'),('User deadline '+$deadline.ToString('o')+'; current optical computation may continue until deadline; no next stage; no automatic resume.'),[Text.UTF8Encoding]::new($false))
Record 'ARMED.json' @{status='ARMED';deadline_utc=$deadline.ToString('o');deadline_japan='2026-11-02 09:00:00 +09:00';guardian_pid=$PID;guardian_creation_utc=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o');owned=$owned;next_stage_inhibited=$true;automatic_resume=$false;code_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLower();utc=[DateTime]::UtcNow.ToString('o')}
$heartbeat=[DateTimeOffset]::MinValue
while([DateTimeOffset]::UtcNow -lt $deadline){
    $now=[DateTimeOffset]::UtcNow
    if(($now-$heartbeat).TotalSeconds -ge 30){Record 'HEARTBEAT.json' @{utc=$now.ToString('o');deadline_utc=$deadline.ToString('o');seconds_remaining=($deadline-$now).TotalSeconds;guardian_pid=$PID};$heartbeat=$now}
    $ms=[Math]::Max(1,[Math]::Min(500,[Math]::Floor(($deadline-[DateTimeOffset]::UtcNow).TotalMilliseconds)))
    Start-Sleep -Milliseconds $ms
}
$trigger=[DateTimeOffset]::UtcNow
$actions=@()
# Stop the actual computation immediately, without waiting for batch completion.
foreach($item in $owned[0..1]){
    $p=SameProcess $item
    if($null -ne $p){
        $at=[DateTimeOffset]::UtcNow
        try{$p.Kill();$actions+=@{pid=$item.pid;role=$item.role;action='terminate_owned_at_user_deadline';requested_utc=$at.ToString('o');success=$true}}
        catch{$actions+=@{pid=$item.pid;role=$item.role;action='terminate_owned_at_user_deadline';requested_utc=$at.ToString('o');success=$false;error=$_.Exception.Message}}
    } else {$actions+=@{pid=$item.pid;role=$item.role;action='original_process_already_absent';utc=[DateTime]::UtcNow.ToString('o')}}
}
[IO.File]::WriteAllText((Join-Path $optical 'REQUEST_STOP'),'User requested stop at 2026-11-02T00:00:00Z. No automatic retry.',[Text.UTF8Encoding]::new($false))
Record 'STOP_TRIGGERED.json' @{status='USER_DEADLINE_STOP';trigger_utc=$trigger.ToString('o');deadline_utc=$deadline.ToString('o');lateness_seconds=($trigger-$deadline).TotalSeconds;actions=$actions;automatic_resume=$false}
# Give only noncomputing wrappers time to flush existing exit/resource receipts.
Start-Sleep -Seconds 5
foreach($item in $owned){
    $p=SameProcess $item
    if($null -ne $p){
        try{$p.Kill();$actions+=@{pid=$item.pid;role=$item.role;action='close_remaining_owned_process';utc=[DateTime]::UtcNow.ToString('o');success=$true}}
        catch{$actions+=@{pid=$item.pid;role=$item.role;action='close_remaining_owned_process';utc=[DateTime]::UtcNow.ToString('o');success=$false;error=$_.Exception.Message}}
    }
}
Start-Sleep -Milliseconds 500
$remaining=@($owned|Where-Object {$null -ne (SameProcess $_)}|ForEach-Object {$_.pid})
$manifest=Get-Content -LiteralPath (Join-Path $base 'formal_optics\MANIFEST.json') -Raw|ConvertFrom-Json
$committed=@();$pending=@()
foreach($job in $manifest.jobs){
    $folder=Split-Path $job.output
    $receipt=Join-Path $folder 'COMPLETE.json'
    if(Test-Path -LiteralPath $receipt){
        $r=Get-Content -LiteralPath $receipt -Raw|ConvertFrom-Json
        if($r.status -eq 'COMPLETE' -and $r.config_sha256 -eq $job.config_sha256 -and $r.source_bank_sha256 -eq $job.bank_sha256 -and (Test-Path -LiteralPath $job.output)){$committed+=$job.job;continue}
    }
    $pending+=@{job=$job.job;frames=$job.count;output=$job.output;partial_exists=(Test-Path -LiteralPath (Join-Path $folder 'CAPTURE.partial.mat'));uncommitted_output_exists=(Test-Path -LiteralPath $job.output);status='NOT_COMMITTED_DO_NOT_CONSUME'}
}
Record 'REMAINING_CAPTURE_JOBS.json' @{committed_jobs=$committed;pending=$pending;no_assets_deleted=$true;no_automatic_resume=$true}
Record 'STOPPED.json' @{status=$(if($remaining.Count -eq 0){'USER_STOPPED'}else{'STOP_INCOMPLETE'});deadline_utc=$deadline.ToString('o');trigger_utc=$trigger.ToString('o');verification_utc=[DateTime]::UtcNow.ToString('o');actions=$actions;remaining_original_pids=$remaining;completed_optical_jobs=$committed.Count;total_optical_jobs=$manifest.jobs.Count;neural_or_fit_started=$false;automatic_resume=$false;classification='Intentional user stop, not a spontaneous crash';preserved_manifest=(Join-Path $base 'formal_optics\MANIFEST.json')}
