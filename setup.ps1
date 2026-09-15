$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Update-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Install-Python {
    if (Test-CommandExists "python") {
        Write-Host "Python already installed: $(python --version)"
        return
    }
    if (Test-CommandExists "py") {
        Write-Host "Python already installed: $(py -3 --version)"
        return
    }

    Write-Host "Python 3 not found; installing it..."
    if (Test-CommandExists "winget") {
        winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    }
    elseif (Test-CommandExists "choco") {
        choco install python312 -y
    }
    else {
        throw "No supported package manager found (winget or Chocolatey). Install Python 3 manually."
    }

    Update-ProcessPath
    if (-not (Test-CommandExists "python") -and -not (Test-CommandExists "py")) {
        throw "Python was installed but is not on PATH. Open a new terminal and run setup.ps1 again."
    }
}

function Install-CCompiler {
    foreach ($compiler in @("cc", "clang", "gcc")) {
        if (Test-CommandExists $compiler) {
            $version = (& $compiler --version) -split "`n" | Select-Object -First 1
            Write-Host "C compiler already installed: $version"
            return
        }
    }

    Write-Host "C compiler not found; installing LLVM/Clang..."
    if (Test-CommandExists "winget") {
        winget install --id LLVM.LLVM -e --accept-source-agreements --accept-package-agreements
    }
    elseif (Test-CommandExists "choco") {
        choco install llvm -y
    }
    else {
        throw "No supported package manager found (winget or Chocolatey). Install LLVM/Clang manually."
    }

    Update-ProcessPath
    if (-not (Test-CommandExists "clang")) {
        throw "LLVM was installed but clang is not on PATH. Open a new terminal and run setup.ps1 again."
    }
}

try {
    Install-Python
    Install-CCompiler

    # Ensure the ext/ dependencies are present.
    if (-not (Test-Path "ext\minifb\include\MiniFB.h")) {
        Write-Host "ext/minifb missing; copying from apps/.lib99/dskbuf..."
        $lib99 = Join-Path (Split-Path $ScriptDir -Parent) ".lib99\dskbuf"
        if (Test-Path $lib99) {
            New-Item -ItemType Directory -Force -Path "ext\minifb" | Out-Null
            Copy-Item -Path "$lib99\*" -Destination "ext\minifb\" -Recurse -Force
        }
        else {
            Write-Host "WARNING: apps/.lib99/dskbuf not found; ext/minifb must be provided manually."
        }
    }

    # Runtime-id output dir (build/<rid>), matching make.py.
    if ($env:OS -like "*Windows*") { $ridOs = "win" }
    elseif ($IsMacOS) { $ridOs = "osx" }
    else { $ridOs = "linux" }
    switch ($env:PROCESSOR_ARCHITECTURE) {
        "AMD64" { $ridArch = "x64" }
        "ARM64" { $ridArch = "arm64" }
        default { $ridArch = "x86" }
    }
    New-Item -ItemType Directory -Force -Path "build\$ridOs-$ridArch", "build\$ridOs-$ridArch\shots", "docs\shots" | Out-Null
    Write-Host "rpg99 build tools are ready."
}
catch {
    Write-Error "Setup failed: $_"
    exit 1
}
finally {
    Pop-Location
}