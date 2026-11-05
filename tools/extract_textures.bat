@echo off
rem Unpacks the game archives that hold textures of the map into build\extracted.
rem Map DLCs are extracted last, as their files take priority over the base game's.
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
for %%D in (balkan_e balkan_w balt east fr greece iberia it north polar) do (
    "%PACKER%" extract "%GAME%\dlc_%%D.scs" -root "%OUT%"
    if errorlevel 1 goto failed
)

echo Done: %OUT%
exit /b 0

:failed
echo Extraction failed, see the messages above.
exit /b 1
