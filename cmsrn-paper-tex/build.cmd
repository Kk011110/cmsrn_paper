@echo off
setlocal
pushd "%~dp0"
where xelatex >nul 2>nul
if errorlevel 1 (
  echo XeLaTeX was not found. Add the TeX Live bin directory to PATH.
  popd
  exit /b 1
)
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto build_error
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto build_error
echo PDF ready: main.pdf
popd
exit /b 0
:build_error
echo Compilation failed. See main.log for details.
popd
exit /b 1
