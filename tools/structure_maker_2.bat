@echo off
setlocal enabledelayedexpansion

set "ROOT=%cd%"
set "OUT=structure.txt"

echo Folder structure (all folders shown; files filtered) > "%OUT%"
echo. >> "%OUT%"

for /d /r "%ROOT%" %%D in (*) do (
    rem Print folder path (always)
    set "rel=%%D"
    set "rel=!rel:%ROOT%=.!"
    echo !rel! >> "%OUT%"

    rem Count .txt files
    set count=0
    for %%F in ("%%D\*.txt") do (
        set /a count+=1
    )

    rem If more than 19 .txt files, skip listing files
    if !count! LEQ 19 (
        rem List all non-image files
        for %%F in ("%%D\*") do (
            set "ext=%%~xF"
            echo !ext! | findstr /i ".png .jpg .jpeg .gif .webp .svg" >nul
            if errorlevel 1 (
                if not "%%~nxF"=="" echo     !rel!\%%~nxF >> "%OUT%"
            )
        )
    ) else (
        echo     [skipped: folder contains !count! txt files] >> "%OUT%"
    )
)

echo Done. Output saved to "%OUT%".
pause
