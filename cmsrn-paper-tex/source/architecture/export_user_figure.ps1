$ErrorActionPreference = 'Stop'
$figureSource = Join-Path $PSScriptRoot 'user-model-architecture.vsdx'
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
    Write-Output ('Exported unmodified user figure; page count: ' + $doc.Pages.Count)
} finally {
    if ($null -ne $doc) { $doc.Close() }
    if ($null -ne $app) { $app.Quit() }
}
