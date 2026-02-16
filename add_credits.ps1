# Script to add clips to a user account
# Usage: .\add_credits.ps1

$WORKER_URL = Read-Host "Enter your Worker URL (e.g., https://your-worker.workers.dev)"
$ADMIN_EMAIL = Read-Host "Enter your admin email"
$ADMIN_PASSWORD = Read-Host "Enter your admin password" -AsSecureString
$ADMIN_PASSWORD_TEXT = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($ADMIN_PASSWORD))

$TARGET_EMAIL = "neopioneer0713@gmail.com"
$CLIPS_TO_ADD = 200
$REASON = "Manual clip grant"

Write-Host "`n1. Logging in as admin..." -ForegroundColor Cyan

# Login to get token
$loginBody = @{
    email = $ADMIN_EMAIL
    password = $ADMIN_PASSWORD_TEXT
} | ConvertTo-Json

try {
    $loginResponse = Invoke-RestMethod -Uri "$WORKER_URL/api/auth/login" -Method POST -Body $loginBody -ContentType "application/json"
    $token = $loginResponse.token
    Write-Host "Login successful" -ForegroundColor Green
} catch {
    Write-Host "Login failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "2. Adding $CLIPS_TO_ADD clips to $TARGET_EMAIL..." -ForegroundColor Cyan

# Grant clips
$grantBody = @{
    target_email = $TARGET_EMAIL
    amount = $CLIPS_TO_ADD
    reason = $REASON
} | ConvertTo-Json

try {
    $headers = @{
        "Authorization" = "Bearer $token"
        "Content-Type" = "application/json"
    }

    $grantResponse = Invoke-RestMethod -Uri "$WORKER_URL/api/admin/grant-clips" -Method POST -Headers $headers -Body $grantBody

    Write-Host "Clips added successfully!" -ForegroundColor Green
    Write-Host "Result:" -ForegroundColor Yellow
    $grantResponse | ConvertTo-Json | Write-Host

} catch {
    Write-Host "Clip grant failed: $_" -ForegroundColor Red
    exit 1
}
