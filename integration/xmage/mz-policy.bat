@echo off
setlocal
if "%~1"=="" exit /b 2
rem Copy this launcher into the root of the rebuilt XMage distribution.
pushd "%~dp0"
java -Xmx8G -Dh2.bindAddress=127.0.0.1 -cp "lib/*" org.mage.magezero.PolicyGames "%~1"
set POLICY_EXIT=%ERRORLEVEL%
popd
exit /b %POLICY_EXIT%
