@echo off
setlocal
cd /d "%~dp0"
"C:\texlive\2026\bin\windows\uplatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 main.tex
if errorlevel 1 exit /b 1
"C:\texlive\2026\bin\windows\uplatex.exe" -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 main.tex
if errorlevel 1 exit /b 1
"C:\texlive\2026\bin\windows\dvipdfmx.exe" -o main.pdf main.dvi
if errorlevel 1 exit /b 1
endlocal
