$ErrorActionPreference = 'Stop'
Add-Type -Path (Join-Path $PSScriptRoot 'Ptz.cs')
while ($null -ne ($line = [Console]::ReadLine())) {
    try {
        $command = $line | ConvertFrom-Json
        if ($command.lensProbe) {
            $result = @{ ok = $true; zoom = [LinkPtz]::ProbeZoom(); focus = [LinkPtz]::ProbeFocus() }
            [Console]::WriteLine(($result | ConvertTo-Json -Compress))
            continue
        }
        elseif ($null -ne $command.zoomSteps) {
            $zoom = [LinkPtz]::AdjustZoom($command.zoomSteps)
            if ($null -eq $zoom) { throw 'Zoom readback failed.' }
            [Console]::WriteLine((@{ ok = $true; zoom = $zoom } | ConvertTo-Json -Compress))
            continue
        }
        elseif ($command.probe) { [LinkPtz]::Probe() }
        else { [LinkPtz]::Move($command.pan, $command.tilt) }
        [Console]::WriteLine('{"ok":true}')
    } catch {
        [Console]::WriteLine((@{ ok = $false; error = $_.Exception.GetBaseException().Message } | ConvertTo-Json -Compress))
    }
}
