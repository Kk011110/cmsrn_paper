param([string]$PythonExecutable = 'python')
$ErrorActionPreference = 'Stop'
$figureSource = Join-Path $PSScriptRoot 'formatted-model-architecture.vsdx'
$figureDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../figures'))
$app = $null
$doc = $null
try {
    $app = New-Object -ComObject Visio.InvisibleApp
    $app.AlertResponse = 7
    $doc = $app.Documents.OpenEx($figureSource, 450)
    $page = $doc.Pages.Item(1)
    $doc.ExportAsFixedFormat(1, (Join-Path $figureDir 'model-architecture.pdf'), 1, 0)
    $page.Export((Join-Path $figureDir 'model-architecture.svg'))
    Write-Output ('Exported formatted user figure; page count: ' + $doc.Pages.Count)
} finally {
    if ($null -ne $doc) { $doc.Close() }
    if ($null -ne $app) { $app.Quit() }
}

& $PythonExecutable (Join-Path $PSScriptRoot 'finish_figure_export.py')
if ($LASTEXITCODE -ne 0) { throw 'Figure PDF cleanup failed' }

& $PythonExecutable (Join-Path $PSScriptRoot 'place_panel_titles_below.py') --refresh-input
if ($LASTEXITCODE -ne 0) { throw 'Figure panel-title layout failed' }
