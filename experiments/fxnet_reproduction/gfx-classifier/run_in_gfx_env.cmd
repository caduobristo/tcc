@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run_in_gfx_env.ps1" %*
exit /b %errorlevel%
