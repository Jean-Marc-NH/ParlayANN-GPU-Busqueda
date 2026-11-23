@echo off
rem ============================================================================
rem COMPARACION 1 COMPLETA (busqueda CPU vs GPU) DE UNA SOLA VEZ
rem
rem ANTES hay que tener la linea base de CPU en WSL (carpeta cpu_baseline):
rem     bash setup.sh
rem     bash run_benchmark.sh msspacev-1M
rem
rem uso:  comparacion1.bat [carpeta_data] [csv_cpu]
rem   sin argumentos se detecta todo solo:
rem     carpeta_data : ~/parlay/big-ann-benchmarks/data dentro de WSL
rem     csv_cpu      : el all_results.csv mas reciente de ..\cpu_baseline\results
rem
rem pasos:
rem   0. guardar los datos del equipo (CPU, RAM, plan de energia, GPU)
rem   1. compilar todo
rem   2. paso1_carga.exe   -> verificar que datos y grafos llegan bien a la GPU
rem   3. comparacion1.exe  -> 4 algoritmos x 18 configuraciones, mejor de 5
rem   4. reporte_comparacion1.py (en WSL) -> validacion vecino por vecino,
rem      tablas y graficas en resultados\comparacion1\
rem ============================================================================
setlocal
cd /d "%~dp0"

rem nombre de la distro de WSL donde esta ~/parlay (cambiar si es otra)
set "DISTRO=Ubuntu"

set "CARPETA=%~1"
set "CSVCPU=%~2"

rem ---- carpeta de datos: se le pregunta a WSL donde esta (sirve con cualquier usuario)
if "%CARPETA%"=="" (
    for /f "delims=" %%i in ('wsl.exe -d %DISTRO% -- bash -c "wslpath -w ~/parlay/big-ann-benchmarks/data"') do set "CARPETA=%%i"
)
if "%CARPETA%"=="" (
    echo ERROR: no encuentro ~/parlay/big-ann-benchmarks/data en WSL. Correr antes setup.sh
    exit /b 1
)

rem ---- csv de la CPU: el mas reciente de cpu_baseline\results (dir /o:d = por fecha)
if "%CSVCPU%"=="" (
    for /f "delims=" %%d in ('dir /b /ad /o:d "..\cpu_baseline\results\msspacev-1M_t*" 2^>nul') do (
        if exist "..\cpu_baseline\results\%%d\all_results.csv" set "CSVCPU=..\cpu_baseline\results\%%d\all_results.csv"
    )
)
if "%CSVCPU%"=="" set "CSVCPU=-"

echo Datos:   %CARPETA%
echo CSV CPU: %CSVCPU%

echo.
echo [0/4] Guardando datos del equipo en resultados\comparacion1_equipo_windows.txt ...
if not exist resultados mkdir resultados
(
    echo === CPU ===
    powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"
    echo === RAM ===
    powershell -NoProfile -Command "Get-CimInstance Win32_PhysicalMemory | Format-Table Manufacturer,Capacity,Speed,ConfiguredClockSpeed -AutoSize"
    echo === PLAN DE ENERGIA ===
    powercfg /getactivescheme
    echo.
    echo === MODO DE ENERGIA DE WINDOWS 11 (el que importa^) ===
    echo ded574b5-... = Mejor rendimiento, 961cc777-... = Mejor eficiencia, 00000000-... = Equilibrado
    reg query "HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" /v ActiveOverlayAcPowerScheme
    echo === GPU ===
    nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max --format=csv
) > resultados\comparacion1_equipo_windows.txt 2>&1

echo.
echo [1/4] Compilando ...
rem OJO: todo con ruta completa (%~dp0). En algunas PCs cmd NO busca programas en la
rem carpeta actual (NoDefaultCurrentDirectoryInExePath) y "build.bat" solo no se encuentra
call "%~dp0build.bat"
if errorlevel 1 (
    echo ERROR compilando
    exit /b 1
)

echo.
echo [2/4] Verificando datos y grafos en la GPU (paso1_carga.exe) ...
"%~dp0paso1_carga.exe" "%CARPETA%"
if errorlevel 1 (
    echo ERROR en paso1_carga: revisar la carpeta de datos
    exit /b 1
)

echo.
echo [3/4] Midiendo la busqueda en GPU (4 algoritmos x 18 configuraciones, mejor de 5) ...
"%~dp0comparacion1.exe" "%CARPETA%" "%CSVCPU%" - - 5
if errorlevel 1 (
    echo ERROR en comparacion1.exe
    exit /b 1
)

echo.
echo [4/4] Validacion vecino por vecino y reporte (WSL) ...
for /f "delims=" %%i in ('wsl.exe -d %DISTRO% wslpath -a "%CD%"') do set "DIR_WSL=%%i"
wsl.exe -d %DISTRO% -- bash -c "source ~/parlay/venv/bin/activate && cd '%DIR_WSL%' && python reporte_comparacion1.py"
if errorlevel 1 (
    echo ERROR en reporte_comparacion1.py
    exit /b 1
)

echo.
echo Listo. Ver resultados\comparacion1\resumen.md y las graficas en la misma carpeta.
endlocal
