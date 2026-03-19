param(
    [string]$Root = ".",
    [string]$Output = "project_dump.txt",
    [int]$MaxFileSizeKB = 1024
)

$ErrorActionPreference = "SilentlyContinue"

$RootPath = (Resolve-Path $Root).Path
$OutPath = Join-Path $RootPath $Output

# Extensiones binarias / pesadas a omitir por defecto
$SkipExtensions = @(
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".pdf",
    ".zip", ".7z", ".rar", ".gz", ".tar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o",
    ".pt", ".pth", ".onnx", ".ckpt", ".h5", ".pkl", ".pickle",
    ".mat", ".npz", ".npy",
    ".mp3", ".mp4", ".wav", ".avi", ".mov"
)

# Carpetas típicas a omitir
$SkipDirs = @(
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".idea", ".vscode", "build", "dist", "out"
)

function Write-Line {
    param([System.IO.StreamWriter]$Writer, [string]$Text)
    $Writer.WriteLine($Text)
}

function Is-SkippedDir {
    param([string]$FullName)
    foreach ($d in $SkipDirs) {
        if ($FullName -match "(\\|/)$([regex]::Escape($d))(\\|/|$)") { return $true }
    }
    return $false
}

function Get-RelativePath {
    param([string]$Base, [string]$Full)
    return [System.IO.Path]::GetRelativePath($Base, $Full)
}

function Write-Tree {
    param(
        [System.IO.StreamWriter]$Writer,
        [string]$Base,
        [string]$Current,
        [string]$Indent = ""
    )

    $items = Get-ChildItem -LiteralPath $Current -Force | Sort-Object PSIsContainer, Name
    $items = $items | Where-Object { -not (Is-SkippedDir $_.FullName) }

    for ($i = 0; $i -lt $items.Count; $i++) {
        $item = $items[$i]
        $isLast = ($i -eq $items.Count - 1)
        $branch = if ($isLast) { "\-- " } else { "+-- " }

        if ($item.PSIsContainer) {
            Write-Line $Writer "$Indent$branch$($item.Name)\"
            $nextIndent = if ($isLast) { "$Indent    " } else { "$Indent|   " }
            Write-Tree -Writer $Writer -Base $Base -Current $item.FullName -Indent $nextIndent
        } else {
            Write-Line $Writer "$Indent$branch$($item.Name)"
        }
    }
}

function Dump-FileContent {
    param(
        [System.IO.StreamWriter]$Writer,
        [System.IO.FileInfo]$File,
        [string]$Base,
        [int]$MaxKB
    )

    $rel = Get-RelativePath -Base $Base -Full $File.FullName
    $ext = $File.Extension.ToLowerInvariant()
    $sizeKB = [math]::Round($File.Length / 1KB, 2)

    Write-Line $Writer ""
    Write-Line $Writer ("=" * 120)
    Write-Line $Writer "FILE: $rel"
    Write-Line $Writer "SIZE_KB: $sizeKB"
    Write-Line $Writer "LAST_WRITE: $($File.LastWriteTime)"
    Write-Line $Writer ("=" * 120)

    if ($SkipExtensions -contains $ext) {
        Write-Line $Writer "[SKIPPED_BINARY_OR_HEAVY_EXTENSION: $ext]"
        return
    }

    if ($File.Length -gt ($MaxKB * 1KB)) {
        Write-Line $Writer "[SKIPPED_TOO_LARGE: $sizeKB KB > $MaxKB KB]"
        return
    }

    try {
        $content = Get-Content -LiteralPath $File.FullName -Raw -Encoding UTF8
        if ($null -eq $content) {
            Write-Line $Writer "[EMPTY_OR_UNREADABLE]"
            return
        }
        Write-Line $Writer $content
    }
    catch {
        try {
            $content = Get-Content -LiteralPath $File.FullName -Raw -Encoding Default
            if ($null -eq $content) {
                Write-Line $Writer "[EMPTY_OR_UNREADABLE]"
                return
            }
            Write-Line $Writer $content
        }
        catch {
            Write-Line $Writer "[FAILED_TO_READ_TEXT_CONTENT]"
        }
    }
}

$writer = New-Object System.IO.StreamWriter($OutPath, $false, [System.Text.Encoding]::UTF8)

try {
    Write-Line $writer "PROJECT DUMP"
    Write-Line $writer "ROOT: $RootPath"
    Write-Line $writer "GENERATED: $(Get-Date)"
    Write-Line $writer "MAX_FILE_SIZE_KB_FOR_CONTENT: $MaxFileSizeKB"
    Write-Line $writer ""

    Write-Line $writer ("#" * 120)
    Write-Line $writer "# DIRECTORY TREE"
    Write-Line $writer ("#" * 120)
    Write-Line $writer (Split-Path $RootPath -Leaf)
    Write-Tree -Writer $writer -Base $RootPath -Current $RootPath

    Write-Line $writer ""
    Write-Line $writer ("#" * 120)
    Write-Line $writer "# FILE CONTENTS"
    Write-Line $writer ("#" * 120)

    $files = Get-ChildItem -LiteralPath $RootPath -Recurse -File -Force |
        Where-Object { -not (Is-SkippedDir $_.FullName) } |
        Sort-Object FullName

    foreach ($file in $files) {
        Dump-FileContent -Writer $writer -File $file -Base $RootPath -MaxKB $MaxFileSizeKB
    }
}
finally {
    $writer.Close()
}

Write-Host "Done. Dump created at: $OutPath"