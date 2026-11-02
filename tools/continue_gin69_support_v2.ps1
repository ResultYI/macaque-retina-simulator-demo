param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$root='D:\macaque retinal simulator'
$base=Join-Path $root 'reports\GIN_69_6_HC2_BC1_SUPPORT_V2'
$pwsh='C:\Users\win11-pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
if(Test-Path -LiteralPath $RunDirectory){throw 'Existing controller receipt retained'}
New-Item -ItemType Directory -Path $RunDirectory|Out-Null
$lock=[IO.File]::Open((Join-Path $base 'PIPELINE.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$watch=[Diagnostics.Stopwatch]::StartNew();$status='FAILED';$phase='gate'
function Record($path,$value){
    $temporary=$path+'.tmp'
    [IO.File]::WriteAllText($temporary,($value|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $path -Force
}
function Progress($stage,$extra){
    Record (Join-Path $RunDirectory 'PROGRESS.json') @{stage=$stage;detail=$extra;pid=$PID;utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$watch.Elapsed.TotalSeconds}
    Write-Output ($stage+' '+$extra)
}
function Stage($name,$folder){
    if(Test-Path -LiteralPath (Join-Path $RunDirectory 'REQUEST_STOP')){throw 'Explicit pause between complete stages'}
    $script:phase=$name
    $out=Join-Path $base ('executions\'+$folder)
    if(Test-Path -LiteralPath $out){
        $t=Get-Content -LiteralPath (Join-Path $out 'TERMINAL.json') -Raw|ConvertFrom-Json
        $r=Get-Content -LiteralPath (Join-Path $out 'RESOURCE.json') -Raw|ConvertFrom-Json
        if($t.status -ne 'EXITED_0' -or $r.status -ne 'PASS'){throw ('Existing failed/incomplete stage requires explicit repair: '+$folder)}
        Progress 'reuse_clean_stage' $folder
        return
    }
    Progress $name $out
    & $pwsh -NoProfile -File (Join-Path $root 'tools\capture_gin_hc2_bc1_support_v2.ps1') -Stage $name -OutputDirectory $out
    $t=Get-Content -LiteralPath (Join-Path $out 'TERMINAL.json') -Raw|ConvertFrom-Json
    $r=Get-Content -LiteralPath (Join-Path $out 'RESOURCE.json') -Raw|ConvertFrom-Json
    if($t.status -ne 'EXITED_0' -or $r.status -ne 'PASS'){throw ('Stopped after failure; no automatic retry: '+$folder)}
}
try {
    Record (Join-Path $RunDirectory 'CONTROLLER.json') @{pid=$PID;creation=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o');code_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLower();scope='69#6 only: new background, short check,0..45s cache,train0..30,dev30..45 once;45..60sealed';automatic_retry=$false}
    $opt=Get-Content -LiteralPath (Join-Path $base 'optical\MATLAB_RESOURCE.json') -Raw|ConvertFrom-Json
    if($opt.status -ne 'PASS'){throw 'New optical preparation has not passed'}
    Stage 'initialize' 'initialize_01'
    Stage 'short_forward' 'short_forward_01'
    $gate=Get-Content -LiteralPath (Join-Path $base 'SHORT_NUMERICAL_CHECK.json') -Raw|ConvertFrom-Json
    if(-not $gate.passed){throw 'Original short numerical gate failed'}
    Stage 'prepare_formal_optical' 'prepare_formal_optical_01'
    $phase='formal_optics';$out=Join-Path $base 'executions\formal_optics_45_01'
    if(-not(Test-Path -LiteralPath $out)){
        Progress $phase '0..45s source input only, sealed45..60 excluded'
        & $pwsh -NoProfile -File (Join-Path $root 'tools\capture_gin69_formal_optics_support_v2.ps1') -PrefixSeconds 45 -OutputDirectory $out
    }
    $r=Get-Content -LiteralPath (Join-Path $out 'MATLAB_RESOURCE.json') -Raw|ConvertFrom-Json
    if($r.status -ne 'PASS'){throw 'Formal optical capture not complete; existing batches retained'}
    Stage 'prepare_native_prefix' 'prepare_native_prefix_01'
    for($i=0;$i -lt 90;$i++){
        $folder=('formal_segment_{0:D3}_01' -f $i)
        Record (Join-Path $base 'FORMAL_SEGMENT_REQUEST.json') @{index=$i;output=(Join-Path $base ('executions\'+$folder));controller_pid=$PID}
        Stage 'formal_segment' $folder
        if($i -eq 59){Stage 'train' 'train_01'}
    }
    Stage 'evaluate' 'evaluate_01'
    $status='COMPLETE';Progress 'COMPLETE' '69#6 fixed development saved; no extra cell queue'
} catch {
    Record (Join-Path $RunDirectory 'FAILURE.json') @{phase=$phase;error=$_.Exception.Message;stack=$_.ScriptStackTrace;utc=[DateTime]::UtcNow.ToString('o');automatic_retry=$false}
    throw
} finally {
    Record (Join-Path $RunDirectory 'TERMINAL.json') @{status=$status;phase=$phase;wall_seconds=$watch.Elapsed.TotalSeconds;utc=[DateTime]::UtcNow.ToString('o');automatic_retry=$false}
    $lock.Dispose()
}
