@echo off
setlocal enabledelayedexpansion

set "ROOT=%cd%"
set "OUT=structure.txt"

echo Folder structure (all folders + per-extension file counts, max 30 files listed) > "%OUT%"
echo. >> "%OUT%"

for /d /r "%ROOT%" %%D in (*) do (
    rem Compute relative folder path
    set "rel=%%D"
    set "rel=!rel:%ROOT%=.!"
    echo !rel! >> "%OUT%"

    rem Reset counters
    set "extlist="
    for /f "tokens=1 delims==" %%X in ('set count_ 2^>nul') do set "%%X="
    set totalFiles=0
    set printedFiles=0

    rem Count and list (up to 30)
    for %%F in ("%%D\*") do (
        set "attrs=%%~aF"
        rem Only count files
        if /i "!attrs:~0,1!" NEQ "d" (
            set /a totalFiles+=1

            rem Prepare extension
            set "ext=%%~xF"
            if "!ext!"=="" (
                set "ext=noext"
            ) else (
                set "ext=!ext:~1!"
            )

            rem Normalize extension to lowercase
            for %%A in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
                set "ext=!ext:%%A=%%A!"
            )

            rem Increment extension counter
            set /a count_!ext!+=1

            rem Track unique extension list
            echo !extlist! | findstr /i "\<!ext!\>" >nul
            if errorlevel 1 set "extlist=!extlist! !ext!"

            rem Print only the first 30 files
            if !printedFiles! LSS 30 (
                echo     !rel!\%%~nxF >> "%OUT%"
                set /a printedFiles+=1
            )
        )
    )

    rem If more than 30 files, notify
    if !totalFiles! GTR 30 (
        echo     [skipped: folder contains !totalFiles! files, showing only first 30] >> "%OUT%"
    )

    rem Print per-extension summary
    if defined extlist (
        echo     [file type counts:] >> "%OUT%"
        for %%E in (!extlist!) do (
            for /f "tokens=2 delims==" %%C in ('set count_%%E 2^>nul') do (
                echo         .%%E: %%C >> "%OUT%"
            )
        )
    ) else (
        echo     [no files in this folder] >> "%OUT%"
    )

    echo. >> "%OUT%"
)

echo Done. Output saved to "%OUT%".
pause
