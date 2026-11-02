$ErrorActionPreference='Stop'
$root='D:\macaque retinal simulator';$out=$root+'\reports\GIN_69_6_HC2_BC1_SUPPORT_V2\optical'
if(Test-Path -LiteralPath ($out+'\MATLAB_PROCESS.json')){throw 'Existing optical attempt retained'}
Add-Type @'
using System;using System.Runtime.InteropServices;
public static class GINMemory {
 [StructLayout(LayoutKind.Sequential)] public class Status {public uint length=64,load;public ulong total,available,page,availablePage,virtualTotal,virtualAvailable,extended;}
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GlobalMemoryStatusEx([In,Out]Status s);
 public static ulong Available(){var s=new Status();if(!GlobalMemoryStatusEx(s))throw new Exception("memory query failed");return s.available;}
}
'@
$batch="addpath('D:/macaque retinal simulator/tools');f=fopen('D:/macaque retinal simulator/reports/GIN_69_6_HC2_BC1_SUPPORT_V2/optical/MATLAB_SELF.json','w');fprintf(f,'%s',jsonencode(struct('pid',feature('getpid'))));fclose(f);c='D:/macaque retinal simulator/reports/GIN_69_6_HC2_BC1_SUPPORT_V2/optical/CONFIG.json';t=tic;prepare_gin_crt_optics_v1(c);cacheSeconds=toc(t);t=tic;export_gin_crt_tile_v1(c,'D:/macaque retinal simulator/reports/GIN_69_6_HC2_BC1_SUPPORT_V2/optical/CAPTURE.mat');captureSeconds=toc(t);f=fopen('D:/macaque retinal simulator/reports/GIN_69_6_HC2_BC1_SUPPORT_V2/optical/MATLAB_TIMING.json','w');fprintf(f,'%s',jsonencode(struct('cache_seconds',cacheSeconds,'capture_seconds',captureSeconds)));fclose(f);"
$watch=[Diagnostics.Stopwatch]::StartNew();$peak=0L;$availableMin=[ulong]::MaxValue;$status='FAILED';$p=$null;$actual=$null
try {
 if([GINMemory]::Available() -lt 8GB){throw 'Insufficient system memory reserve'}
 $p=Start-Process -FilePath 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe' -ArgumentList @('-batch',('"'+$batch+'"')) -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput ($out+'\MATLAB_STDOUT.log') -RedirectStandardError ($out+'\MATLAB_STDERR.log')
 @{pid=$p.Id;creation=$p.StartTime.ToUniversalTime().ToString('o');start_utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath ($out+'\MATLAB_PROCESS.json') -Encoding utf8
 while(-not $p.HasExited){
   $p.Refresh()
   if($null -eq $actual -and (Test-Path -LiteralPath ($out+'\MATLAB_SELF.json'))){
      $owned=Get-Content -LiteralPath ($out+'\MATLAB_SELF.json') -Raw|ConvertFrom-Json
      $actual=Get-Process -Id $owned.pid
      if($actual.Path -notlike 'C:\Program Files\MATLAB\R2025b\bin\*' -or $actual.StartTime -lt $p.StartTime.AddSeconds(-1)){throw 'MATLAB child identity mismatch'}
   }
   $usage=$p.WorkingSet64
   if($null -ne $actual){$actual.Refresh();if(-not $actual.HasExited){$usage=$actual.WorkingSet64}}
   $peak=[Math]::Max($peak,$usage);$available=[GINMemory]::Available();$availableMin=[Math]::Min($availableMin,$available)
   if($usage -gt 24GB -or $available -lt 8GB -or (Test-Path -LiteralPath ($out+'\REQUEST_STOP'))){
      if($null -ne $actual -and -not $actual.HasExited){Stop-Process -Id $actual.Id};if(-not $p.HasExited){Stop-Process -Id $p.Id};throw 'Owned MATLAB resource/explicit stop guard'
   }
   Start-Sleep -Seconds 2
 }
 $p.WaitForExit();if($p.ExitCode -ne 0){throw ('MATLAB exit '+$p.ExitCode)}
 if(-not(Test-Path -LiteralPath ($out+'\CAPTURE.mat'))){throw 'MATLAB capture output absent'}
 $status='PASS'
} finally {
 @{status=$status;wall_seconds=$watch.Elapsed.TotalSeconds;peak_rss_bytes=$peak;actual_matlab_child_monitored=($null -ne $actual);minimum_available_bytes=$availableMin;limit_bytes=24GB;reserve_bytes=8GB;exit_code=if($null -eq $p){$null}else{$p.ExitCode};automatic_retry=$false}|ConvertTo-Json|Set-Content -LiteralPath ($out+'\MATLAB_RESOURCE.json') -Encoding utf8
}
