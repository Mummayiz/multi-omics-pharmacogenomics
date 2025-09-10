# Multi-Omics Pharmacogenomics Platform - Deployment Monitoring Script
# PowerShell script to monitor deployment health and performance

param(
    [string]$Mode = "all",  # all, health, performance, logs
    [int]$Interval = 30,    # Monitoring interval in seconds
    [switch]$Continuous = $false,
    [switch]$Alert = $false,
    [string]$LogFile = "deployment-monitor.log"
)

Write-Host "🔍 Multi-Omics Platform - Deployment Monitor" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

$ErrorActionPreference = "Continue"

# Configuration
$services = @(
    @{ Name = "API"; URL = "http://localhost:8000/health"; Container = "multi-omics-api" },
    @{ Name = "Frontend"; URL = "http://localhost"; Container = "multi-omics-nginx" },
    @{ Name = "PostgreSQL"; URL = "localhost:5432"; Container = "multi-omics-postgres" },
    @{ Name = "Redis"; URL = "localhost:6379"; Container = "multi-omics-redis" },
    @{ Name = "Jupyter"; URL = "http://localhost:8888"; Container = "multi-omics-jupyter" }
)

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $LogFile -Value $logEntry
}

# Health check function
function Test-ServiceHealth {
    Write-Host "🏥 Health Check Results:" -ForegroundColor Blue
    Write-Host "========================" -ForegroundColor Blue
    
    $healthResults = @()
    
    foreach ($service in $services) {
        $status = "❌ DOWN"
        $responseTime = "N/A"
        $details = ""
        
        try {
            if ($service.Name -eq "API") {
                $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
                $response = Invoke-RestMethod -Uri $service.URL -TimeoutSec 5
                $stopwatch.Stop()
                
                if ($response.status -eq "healthy") {
                    $status = "✅ UP"
                    $responseTime = "$($stopwatch.ElapsedMilliseconds)ms"
                    $details = "Services: $($response.services | ConvertTo-Json -Compress)"
                } else {
                    $status = "⚠️  DEGRADED"
                    $details = "Status: $($response.status)"
                }
            }
            elseif ($service.Name -eq "Frontend") {
                $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
                $response = Invoke-WebRequest -Uri $service.URL -TimeoutSec 5
                $stopwatch.Stop()
                
                if ($response.StatusCode -eq 200) {
                    $status = "✅ UP"
                    $responseTime = "$($stopwatch.ElapsedMilliseconds)ms"
                }
            }
            elseif ($service.Name -eq "PostgreSQL") {
                $containerStatus = docker ps --filter "name=$($service.Container)" --format "{{.Status}}"
                if ($containerStatus -like "*Up*") {
                    $status = "✅ UP"
                    $details = $containerStatus
                }
            }
            elseif ($service.Name -eq "Redis") {
                $containerStatus = docker ps --filter "name=$($service.Container)" --format "{{.Status}}"
                if ($containerStatus -like "*Up*") {
                    $status = "✅ UP"
                    $details = $containerStatus
                }
            }
            elseif ($service.Name -eq "Jupyter") {
                try {
                    $response = Invoke-WebRequest -Uri $service.URL -TimeoutSec 5
                    if ($response.StatusCode -eq 200) {
                        $status = "✅ UP"
                    }
                } catch {
                    # Jupyter might require authentication, so 401/403 is also "up"
                    if ($_.Exception.Response.StatusCode -eq 401 -or $_.Exception.Response.StatusCode -eq 403) {
                        $status = "✅ UP (Auth Required)"
                    }
                }
            }
        }
        catch {
            $status = "❌ DOWN"
            $details = $_.Exception.Message
        }
        
        $result = @{
            Service = $service.Name
            Status = $status
            ResponseTime = $responseTime
            Details = $details
            Timestamp = Get-Date
        }
        
        $healthResults += $result
        
        Write-Host "   $($service.Name.PadRight(12)) : $status ($responseTime)" -ForegroundColor $(
            if ($status.StartsWith("✅")) { "Green" }
            elseif ($status.StartsWith("⚠️")) { "Yellow" }
            else { "Red" }
        )
        
        if ($details) {
            Write-Host "                    Details: $details" -ForegroundColor Gray
        }
    }
    
    Write-Log "Health check completed. Results: $(($healthResults | Where-Object { $_.Status.StartsWith('✅') }).Count)/$($healthResults.Count) services healthy"
    
    return $healthResults
}

# Performance monitoring function
function Test-Performance {
    Write-Host "📊 Performance Metrics:" -ForegroundColor Blue
    Write-Host "=======================" -ForegroundColor Blue
    
    # Docker container stats
    try {
        Write-Host "   Container Resource Usage:" -ForegroundColor Gray
        $containerStats = docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        $containerStats | ForEach-Object { Write-Host "     $_" -ForegroundColor Gray }
    }
    catch {
        Write-Host "   ❌ Could not retrieve container stats" -ForegroundColor Red
    }
    
    # API performance test
    try {
        Write-Host "   API Performance Test:" -ForegroundColor Gray
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
        $stopwatch.Stop()
        
        Write-Host "     Response Time: $($stopwatch.ElapsedMilliseconds)ms" -ForegroundColor Gray
        Write-Host "     Status: $($response.status)" -ForegroundColor Gray
    }
    catch {
        Write-Host "     ❌ API performance test failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # System resources
    try {
        Write-Host "   System Resources:" -ForegroundColor Gray
        $cpu = Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average
        $memory = Get-WmiObject -Class Win32_OperatingSystem
        $memoryUsed = [math]::Round((($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / $memory.TotalVisibleMemorySize) * 100, 2)
        
        Write-Host "     CPU Usage: $($cpu.Average)%" -ForegroundColor Gray
        Write-Host "     Memory Usage: $memoryUsed%" -ForegroundColor Gray
    }
    catch {
        Write-Host "     ⚠️  Could not retrieve system resources" -ForegroundColor Yellow
    }
    
    Write-Log "Performance monitoring completed"
}

# Log monitoring function
function Watch-Logs {
    Write-Host "📋 Recent Container Logs:" -ForegroundColor Blue
    Write-Host "=========================" -ForegroundColor Blue
    
    $containers = @("multi-omics-api", "multi-omics-postgres", "multi-omics-redis", "multi-omics-nginx")
    
    foreach ($container in $containers) {
        try {
            Write-Host "   📋 $container logs (last 5 lines):" -ForegroundColor Gray
            $logs = docker logs --tail 5 $container 2>&1
            $logs | ForEach-Object { 
                $logLine = $_.ToString()
                $color = "Gray"
                if ($logLine -match "ERROR|FATAL|Exception") { $color = "Red" }
                elseif ($logLine -match "WARN|WARNING") { $color = "Yellow" }
                elseif ($logLine -match "INFO") { $color = "White" }
                
                Write-Host "     $logLine" -ForegroundColor $color
            }
            Write-Host ""
        }
        catch {
            Write-Host "     ❌ Could not retrieve logs for $container" -ForegroundColor Red
        }
    }
    
    Write-Log "Log monitoring completed"
}

# Alert function
function Send-Alert {
    param([string]$Message, [string]$Severity = "WARNING")
    
    Write-Host "🚨 ALERT [$Severity]: $Message" -ForegroundColor Red
    Write-Log "ALERT [$Severity]: $Message" "ALERT"
    
    # Here you could add integration with external alerting systems
    # - Email notifications
    # - Slack webhooks
    # - PagerDuty
    # - etc.
}

# Main monitoring loop
function Start-Monitoring {
    Write-Log "Starting deployment monitoring (Mode: $Mode, Interval: ${Interval}s, Continuous: $Continuous)"
    
    do {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host ""
        Write-Host "🔍 Monitoring Report - $timestamp" -ForegroundColor Cyan
        Write-Host "================================================" -ForegroundColor Cyan
        
        # Health checks
        if ($Mode -eq "all" -or $Mode -eq "health") {
            $healthResults = Test-ServiceHealth
            
            # Check for failures and send alerts if enabled
            if ($Alert) {
                $failedServices = $healthResults | Where-Object { $_.Status.StartsWith("❌") }
                foreach ($failed in $failedServices) {
                    Send-Alert "Service $($failed.Service) is DOWN: $($failed.Details)"
                }
            }
        }
        
        # Performance monitoring
        if ($Mode -eq "all" -or $Mode -eq "performance") {
            Write-Host ""
            Test-Performance
        }
        
        # Log monitoring
        if ($Mode -eq "all" -or $Mode -eq "logs") {
            Write-Host ""
            Watch-Logs
        }
        
        if ($Continuous) {
            Write-Host ""
            Write-Host "⏳ Waiting $Interval seconds until next check..." -ForegroundColor Yellow
            Write-Host "   Press Ctrl+C to stop monitoring" -ForegroundColor Gray
            Start-Sleep -Seconds $Interval
        }
        
    } while ($Continuous)
}

# Script execution
try {
    # Verify Docker is running
    docker version | Out-Null
    
    # Create log file if it doesn't exist
    if (!(Test-Path $LogFile)) {
        New-Item -ItemType File -Path $LogFile -Force | Out-Null
    }
    
    # Start monitoring
    Start-Monitoring
    
}
catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   Make sure Docker is running and services are deployed." -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "🎉 Monitoring completed!" -ForegroundColor Green
Write-Host "   Log file: $LogFile" -ForegroundColor Gray
