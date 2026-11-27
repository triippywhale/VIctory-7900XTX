# Starts the title with the settings this fork is played with (README.md, "How to run it").
#   scripts\play.ps1 -Exe <path of your own copy's executable>
# -Safe         without the newer, faster ways (the first thing to try when something is wrong)
# -RayTracing   the title's own ray-traced indirect light: lit interiors, about a third slower
# -Fsr4K        PC FidelityFX 4.1.2 at 3840x2160 (quality, balanced or performance)
# -Build        target\diag (the default, the build all measurements are of) or target\release
# -More         other settings: -More @{VICTORY_PANEL='1'}
# -Ports        which renderer patches run, for A/B runs (README.md, "Renderer patches"):
#               1 = VICTORY_PORT_CONSTANTS, 2 = VICTORY_PORT_DESCRIPTORS, 5 = VICTORY_PORT_CBV,
#               6 = VICTORY_PORT_GUARD, 7 = VICTORY_PORT_JOURNAL. The listed ones are set to 1 and
#               the others to 0: -Ports 1,5, -Ports all, or -Ports none (all five off). Without it
#               each keeps the build's default (1 and 5 on, the rest off). -More still wins over it.
# -CpuPorts     two recording-thread trims from Op-lite/victory-scarlett, both off without it:
#               VICTORY_AFFINITY=free (the title's five recording threads not pinned to logical
#               processors 8-12) and VICTORY_TABLE_MEMO=1 (a table entry made ready earlier in a list,
#               with nothing changed since, not made ready again). -More @{VICTORY_TABLE_MEMO='verify'}
#               checks the memo instead: its frame line must count 0 changes.
# -CardLists    VICTORY_CARD_LISTS=1: card copies whose CPU side changed come from a list made as
#               they are touched, not a walk over all of them at every submission (off without it)
# -LateLists    the most host lists in the air before the flights are settled
#               (VICTORY_LATE_WAIT_LISTS; the build's default is 256), for A/B runs: -LateLists 1024
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
    [switch]$CpuPorts,
    [switch]$CardLists,
    [int]$LateLists = 0,
    [ValidateSet('quality', 'balanced', 'performance')][string]$Fsr4K,
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [hashtable]$More = @{},
    [switch]$NoPrebuild,
    [switch]$CleanCache,
    [ValidateSet('1', '2', '5', '6', '7', 'all', 'none')][string[]]$Ports = @()
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
if ($CpuPorts) { $settings.VICTORY_AFFINITY = 'free'; $settings.VICTORY_TABLE_MEMO = '1' }
if ($CardLists) { $settings.VICTORY_CARD_LISTS = '1' }
if ($LateLists -gt 0) { $settings.VICTORY_LATE_WAIT_LISTS = "$LateLists" }
$portSwitches = [ordered]@{
    '1' = 'VICTORY_PORT_CONSTANTS'; '2' = 'VICTORY_PORT_DESCRIPTORS'; '5' = 'VICTORY_PORT_CBV'
    '6' = 'VICTORY_PORT_GUARD'; '7' = 'VICTORY_PORT_JOURNAL'
}
if ($Ports) {
    if (($Ports -contains 'none') -and $Ports.Count -gt 1) { Write-Host "-Ports none goes alone."; exit 1 }
    $wanted = if ($Ports -contains 'all') { @($portSwitches.Keys) } else { @($Ports | Where-Object { $_ -ne 'none' }) }
    foreach ($n in $portSwitches.Keys) { $settings[$portSwitches[$n]] = if ($wanted -contains $n) { '1' } else { '0' } }
    $on = if ($wanted) { ($wanted | Sort-Object -Unique) -join ', ' } else { 'none' }
    Write-Host "Renderer patches on: $on (the others set to 0)"
}
if ($Fsr4K) { $settings.VICTORY_FSR_4K = $Fsr4K.ToLowerInvariant() }
foreach ($k in $More.Keys) { $settings[$k] = $More[$k] }

$victory = Join-Path $PSScriptRoot "..\target\$Build\victory.exe"
if (-not (Test-Path $victory)) { Write-Host "No $victory. Build it first (README.md)."; exit 1 }
if (-not (Test-Path $Exe)) { Write-Host "No $Exe."; exit 1 }
# (Other settings left over in this window go; where your files are stays.)
$paths = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR', 'VICTORY_FIDELITYFX_DIR'
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
