Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue
Start-Sleep 2

$p = New-Object System.Diagnostics.ProcessStartInfo
$p.FileName = 'C:\Users\nagar\AppData\Local\Programs\Ollama\ollama.exe'
$p.Arguments = 'serve'
$p.UseShellExecute = $false
$p.EnvironmentVariables['OLLAMA_ORIGINS'] = '*'
$p.EnvironmentVariables['OLLAMA_HOST'] = '0.0.0.0'

$proc = [System.Diagnostics.Process]::Start($p)
Start-Sleep 4
Write-Host "Ollama started with PID $($proc.Id) and CORS open"
