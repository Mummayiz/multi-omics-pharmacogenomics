# Push Multi-Omics Pharmacogenomics Platform to GitHub
# PowerShell script for Windows

# Colors for output
$Green = [System.ConsoleColor]::Green
$Red = [System.ConsoleColor]::Red
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue

Write-Host "🧬 Multi-Omics Pharmacogenomics Platform - GitHub Setup" -ForegroundColor $Blue
Write-Host "=======================================================" -ForegroundColor $Blue
Write-Host ""

# Get GitHub username
Write-Host "📝 Please enter your GitHub username:" -ForegroundColor $Yellow
$username = Read-Host "GitHub Username"

if ([string]::IsNullOrEmpty($username)) {
    Write-Host "❌ GitHub username is required!" -ForegroundColor $Red
    exit 1
}

# Construct repository URL
$repoUrl = "https://github.com/$username/multi-omics-pharmacogenomics-platform.git"

Write-Host ""
Write-Host "🔧 Setting up repository..." -ForegroundColor $Green
Write-Host "Repository URL: $repoUrl" -ForegroundColor $Blue

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠️  Remote 'origin' already exists: $existingRemote" -ForegroundColor $Yellow
    Write-Host "Do you want to update it? (y/N)" -ForegroundColor $Yellow
    $updateRemote = Read-Host
    
    if ($updateRemote -eq "y" -or $updateRemote -eq "Y") {
        Write-Host "🔄 Updating remote origin..." -ForegroundColor $Green
        git remote set-url origin $repoUrl
    } else {
        Write-Host "❌ Cancelled." -ForegroundColor $Red
        exit 1
    }
} else {
    Write-Host "➕ Adding remote origin..." -ForegroundColor $Green
    git remote add origin $repoUrl
}

# Verify remote was added
$remoteCheck = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to add remote repository!" -ForegroundColor $Red
    exit 1
}

Write-Host "✅ Remote added successfully: $remoteCheck" -ForegroundColor $Green

# Set main branch
Write-Host "🌿 Setting main branch..." -ForegroundColor $Green
git branch -M main

# Push to GitHub
Write-Host "🚀 Pushing to GitHub..." -ForegroundColor $Green
Write-Host "This will upload all your code to: $repoUrl" -ForegroundColor $Blue
Write-Host ""

$pushResult = git push -u origin main 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "🎉 SUCCESS! Repository pushed to GitHub!" -ForegroundColor $Green
    Write-Host "🌐 Your repository is now available at:" -ForegroundColor $Blue
    Write-Host "   https://github.com/$username/multi-omics-pharmacogenomics-platform" -ForegroundColor $Blue
    Write-Host ""
    Write-Host "📊 Repository Statistics:" -ForegroundColor $Green
    $fileCount = (git ls-files).Count
    $commitCount = (git rev-list --count HEAD)
    Write-Host "   Files: $fileCount" -ForegroundColor $Blue
    Write-Host "   Commits: $commitCount" -ForegroundColor $Blue
    Write-Host ""
    Write-Host "🔗 Next Steps:" -ForegroundColor $Yellow
    Write-Host "   1. Visit your repository on GitHub" -ForegroundColor $Blue
    Write-Host "   2. Add a description and topics" -ForegroundColor $Blue
    Write-Host "   3. Consider adding a LICENSE file" -ForegroundColor $Blue
    Write-Host "   4. Set up GitHub Pages for documentation" -ForegroundColor $Blue
} else {
    Write-Host "❌ Failed to push to GitHub!" -ForegroundColor $Red
    Write-Host "Error details:" -ForegroundColor $Red
    Write-Host $pushResult -ForegroundColor $Red
    Write-Host ""
    Write-Host "💡 Common solutions:" -ForegroundColor $Yellow
    Write-Host "   1. Make sure the repository exists on GitHub" -ForegroundColor $Blue
    Write-Host "   2. Check your GitHub username is correct" -ForegroundColor $Blue
    Write-Host "   3. Ensure you're logged in to Git (git config --global user.name)" -ForegroundColor $Blue
    Write-Host "   4. You might need to authenticate with GitHub" -ForegroundColor $Blue
}
