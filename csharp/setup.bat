@echo off

REM Builds NuGet package that wraps TensorRT plugin EP

IF "%~1"=="" (
    echo ERROR: No build configuration specified.
    echo Usage: .\setup.bat [Debug^|Release]
    exit /b 1
)

SET "BUILD_CONFIG=%~1"

if "%TENSORRT_PLUGIN_EP_LIBRARY_PATH%"=="" (
    echo ERROR: TENSORRT_PLUGIN_EP_LIBRARY_PATH environment variable is not set.
    exit /b 1
)

if not exist "%TENSORRT_PLUGIN_EP_LIBRARY_PATH%" (
    echo ERROR: EP library "%TENSORRT_PLUGIN_EP_LIBRARY_PATH%" not found.
    exit /b 1
)

set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"

if /i "%ARCH%"=="AMD64" (
    set "DEST_EP_DLL_FOLDER=.\Microsoft.ML.OnnxRuntime.EP.TensorRT\runtimes\win-x64\native\"
) else if /i "%ARCH%"=="ARM64" (
    set "DEST_EP_DLL_FOLDER=.\Microsoft.ML.OnnxRuntime.EP.TensorRT\runtimes\win-arm64\native\"
) else (
    echo ERROR: Unknown architecture "%ARCH%"
    exit /b 1
)

if not exist "%DEST_EP_DLL_FOLDER%" (
    mkdir "%DEST_EP_DLL_FOLDER%" || (
        echo ERROR: Failed to create "%DEST_EP_DLL_FOLDER%".
        exit /b 1
    )
)

echo Copying EP DLL to "%DEST_EP_DLL_FOLDER%"
copy /Y "%TENSORRT_PLUGIN_EP_LIBRARY_PATH%" "%DEST_EP_DLL_FOLDER%" >nul

if errorlevel 1 (
    echo ERROR: Failed to copy EP library to "%DEST_EP_DLL_FOLDER%".
    exit /b 1
)

echo Building NuGet package ("%BUILD_CONFIG%") ...
dotnet build .\Microsoft.ML.OnnxRuntime.EP.TensorRT\Microsoft.ML.OnnxRuntime.EP.TensorRT.csproj -c "%BUILD_CONFIG%"
dotnet pack .\Microsoft.ML.OnnxRuntime.EP.TensorRT\Microsoft.ML.OnnxRuntime.EP.TensorRT.csproj -c "%BUILD_CONFIG%"

set "LOCAL_FEED_FOLDER=local_feed"
if not exist "%LOCAL_FEED_FOLDER%" (
    mkdir "%LOCAL_FEED_FOLDER%" || (
        echo ERROR: Failed to create "%LOCAL_FEED_FOLDER%"
        exit /b 1
    )
)

copy /Y .\Microsoft.ML.OnnxRuntime.EP.TensorRT\bin\"%BUILD_CONFIG%"\Microsoft.ML.OnnxRuntime.EP.TensorRT.*.nupkg .\local_feed\
copy /Y .\Microsoft.ML.OnnxRuntime.EP.TensorRT\bin\"%BUILD_CONFIG%"\Microsoft.ML.OnnxRuntime.EP.TensorRT.*.snupkg .\local_feed\
