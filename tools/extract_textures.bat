@echo off
rem Unpacks the two game archives that hold the terrain textures into build\extracted.
rem The game itself is only read, nothing in its folder changes.
setlocal
set GAME=D:\SteamLibrary\steamapps\common\Euro Truck Simulator 2
set PACKER=%~dp0scs_packer\scs_packer.exe
set OUT=%~dp0..\build\extracted
if not exist "%OUT%" mkdir "%OUT%"

"%PACKER%" extract "%GAME%\base.scs" -root "%OUT%"
if errorlevel 1 goto failed
"%PACKER%" extract "%GAME%\base_share.scs" -root "%OUT%"
if errorlevel 1 goto failed

echo Done: %OUT%
exit /b 0

:failed
echo Extraction failed, see the messages above.
exit /b 1
