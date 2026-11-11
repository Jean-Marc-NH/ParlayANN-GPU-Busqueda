@echo off
rem compila el proyecto cuda con nvcc + el compilador de visual studio 2022
rem uso: doble click o "build.bat" desde una terminal

cd /d "%~dp0"

rem cargar el entorno de visual studio (cl.exe), nvcc lo necesita en windows
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo No encuentro vcvars64.bat, revisar la ruta de Visual Studio en build.bat
    exit /b 1
)
call "%VCVARS%" >nul

rem se compila para dos GPUs a la vez (requiere CUDA 12.8 o mas nuevo):
rem   sm_86  = RTX 3050 (Ampere)
rem   sm_120 = RTX 5050 (Blackwell)
rem   compute_120 = codigo PTX para que GPUs mas nuevas lo compilen al vuelo
set ARCHS=-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
set FLAGS=-O3 -std=c++17 %ARCHS% -lineinfo -Xcompiler "/W3 /wd4996"

echo Compilando paso1_carga.exe ...
nvcc %FLAGS% -o paso1_carga.exe paso1_carga.cu datos_gpu.cu lectores.cpp
if errorlevel 1 (
    echo ERROR al compilar paso1
    exit /b 1
)

echo Compilando paso2_busqueda.exe ...
nvcc %FLAGS% -o paso2_busqueda.exe paso2_busqueda.cu busqueda.cu datos_gpu.cu lectores.cpp
if errorlevel 1 (
    echo ERROR al compilar paso2
    exit /b 1
)
rem misma busqueda pero midiendo ciclos por seccion del kernel (clock64)
echo Compilando paso2_perfil.exe ...
nvcc %FLAGS% -DPERFIL -o paso2_perfil.exe paso2_busqueda.cu busqueda.cu datos_gpu.cu lectores.cpp
if errorlevel 1 (
    echo ERROR al compilar paso2_perfil
    exit /b 1
)
rem microbenchmark de formas de calcular distancias (justifica la optimizacion)
echo Compilando micro_distancias.exe ...
nvcc %FLAGS% -o micro_distancias.exe micro_distancias.cu datos_gpu.cu lectores.cpp
if errorlevel 1 (
    echo ERROR al compilar micro_distancias
    exit /b 1
)
echo Listo: paso1_carga.exe, paso2_busqueda.exe, paso2_perfil.exe y micro_distancias.exe
