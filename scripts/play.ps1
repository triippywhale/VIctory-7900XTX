# Starts the title with the settings this fork is played with (README.md, "How to run it").
#   scripts\play.ps1 -Exe <path of your own copy's executable>
# -Safe         without the newer, faster ways (the first thing to try when something is wrong)
# -RayTracing   the title's own ray-traced indirect light: lit interiors, about a third slower
# -Build        target\diag (the default, the build all measurements are of) or target\release
# -More         other settings: -More @{VICTORY_PANEL='1'}
# -Ports        renderer patches to turn on, for A/B runs (README.md, "Renderer patches"):
#               1 = VICTORY_PORT_CONSTANTS, 2 = VICTORY_PORT_DESCRIPTORS, 5 = VICTORY_PORT_CBV,
#               6 = VICTORY_PORT_GUARD, 7 = VICTORY_PORT_JOURNAL; e.g. -Ports 1,5 or -Ports all.
#               Without it all five stay off (their default). -More still wins over it.
# -NoPrebuild   skip making the title's pipelines ahead (translate --cache --pipelines), which
#               otherwise runs first, with the same settings: long once per new build or driver,
#               then only what is new
# -CleanCache   before the pre-build, throw away the made pipelines and the translated shaders
#               (<cache>\pipelines\host-pipelines.*, prebuilt-pipelines.txt, warm-done.txt and
#               <cache>\shaders), so that everything is made again from this build. Keeps what
#               only runs can find out: known-*.bin, early-, slow-, crashed-pipelines.txt and
#               warm-bad.txt. For artifacts that look like a stale cache, not for every run: the
#               pre-build after it is a cold one (minutes; far longer with VICTORY_LIBRARY_LOCK=1).
#               The driver's own shader cache is not touched (AMD Software: Reset Shader Cache).
# Run it from a normal window, not as administrator: the driver then keeps its shader cache elsewhere.
param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$Safe,
    [switch]$RayTracing,
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [hashtable]$More = @{},
    [switch]$NoPrebuild,
    [switch]$CleanCache,
    [ValidateSet('1', '2', '5', '6', '7', 'all')][string[]]$Ports = @()
)
$settings = @{
    VICTORY_TEXTURE_NO_WAIT = '1'; VICTORY_RELAY = '2'; VICTORY_WIDE_BACK = '1'; VICTORY_CARD_REHASH_RUNS = '1'
    VICTORY_JOIN_COPIES = '1'; VICTORY_INDIRECT_WINDOW = '1'; VICTORY_RELAY_COPY = '1'; VICTORY_CARD_STEPS = '2'
    VICTORY_INDIRECT_BATCH = '2'; VICTORY_MAP_THREADS = '1'; VICTORY_TITLE_PRIORITIES = '0'; VICTORY_CARD_AWAKE = '1'
    VICTORY_SHOW_AFTER = '1'; VICTORY_AMBIENT = '1'; VICTORY_D3D12_DRED = 'light'
}
if ($Safe) {
    $settings.VICTORY_UNDER_WIDE = '0'; $settings.VICTORY_LAND_ECHO = '0'
} else {
    $settings.VICTORY_THIN = '1'; $settings.VICTORY_JOIN_PARTS = '8'; $settings.VICTORY_UPLOAD_AHEAD = '1'
}
if ($RayTracing) { $settings.VICTORY_BVH_BUILDER = '1'; $settings.VICTORY_BVH_FAULTS = '8192' }
$portSwitches = [ordered]@{
    '1' = 'VICTORY_PORT_CONSTANTS'; '2' = 'VICTORY_PORT_DESCRIPTORS'; '5' = 'VICTORY_PORT_CBV'
    '6' = 'VICTORY_PORT_GUARD'; '7' = 'VICTORY_PORT_JOURNAL'
}
$wanted = if ($Ports -contains 'all') { @($portSwitches.Keys) } else { $Ports }
foreach ($n in $wanted) { $settings[$portSwitches[$n]] = '1' }
if ($wanted) { Write-Host "Renderer patches on: $(($wanted | Sort-Object -Unique) -join ', ')" }
foreach ($k in $More.Keys) { $settings[$k] = $More[$k] }

$victory = Join-Path $PSScriptRoot "..\target\$Build\victory.exe"
if (-not (Test-Path $victory)) { Write-Host "No $victory. Build it first (README.md)."; exit 1 }
if (-not (Test-Path $Exe)) { Write-Host "No $Exe."; exit 1 }
# (Other settings left over in this window go; where your files are stays.)
$paths = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR'
$inherited = @(Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $paths })
foreach ($entry in $inherited) { Remove-Item "Env:$($entry.Name)" }
foreach ($k in $settings.Keys) { Set-Item "Env:$k" $settings[$k] }
$discarded = $inherited | Where-Object {
    $_.Name -notin $More.Keys -and -not $_.Value.Equals([Environment]::GetEnvironmentVariable($_.Name))
} | Sort-Object Name
if ($discarded) {
    Write-Host "Discarded inherited values for $($discarded.Name -join ', '). Use -More to pass them explicitly."
}
$Exe = (Resolve-Path $Exe).Path
Push-Location (Join-Path $PSScriptRoot '..')   # victory.toml is looked for here
try {
    if ($CleanCache) {
        # A running victory.exe has the library mapped and writes to these folders.
        if (Get-Process -Name victory -ErrorAction SilentlyContinue) {
            Write-Host "victory.exe is running; close it before -CleanCache."; exit 1
        }
        $line = & $victory config | Where-Object { $_ -match '^cache_dir\s*:' } | Select-Object -First 1
        if (-not $line) { Write-Host "Could not read cache_dir from 'victory config'."; exit 1 }
        $cache = ($line -replace '^cache_dir\s*:\s*', '').Trim()
        $made = Join-Path $cache 'pipelines'
        $gone = @(
            'host-pipelines.bin', 'host-pipelines.next', 'prebuilt-pipelines.txt', 'warm-done.txt'
        ) | ForEach-Object { Join-Path $made $_ }
        $gone += Join-Path $cache 'shaders'
        $bytes = 0
        foreach ($path in $gone | Where-Object { Test-Path $_ }) {
            $bytes += (Get-ChildItem $path -Recurse -File -Force | Measure-Object Length -Sum).Sum
            Remove-Item $path -Recurse -Force
        }
        Write-Host ("Cleared {0:N0} MB of made pipelines and translated shaders from {1}." -f ($bytes / 1MB), $cache)
        if ($NoPrebuild) { Write-Host "-NoPrebuild: the run itself will make every pipeline it meets (stutter)." }
    }
    if (-not $NoPrebuild) {
        # Every pipeline of the title's shader pack, made before the title needs one (the
        # first loading screen otherwise stutters while the driver makes them).
        & $victory translate --cache --pipelines (Split-Path -Parent $Exe)
        if ($LASTEXITCODE -ne 0) { Write-Host "The pipeline pre-build failed ($LASTEXITCODE); starting anyway." }
    }
    & $victory run $Exe --execute
} finally { Pop-Location }
