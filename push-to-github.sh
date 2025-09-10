#!/bin/bash
# Push Multi-Omics Pharmacogenomics Platform to GitHub
# Bash script for cross-platform compatibility

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧬 Multi-Omics Pharmacogenomics Platform - GitHub Setup${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

# Get GitHub username
echo -e "${YELLOW}📝 Please enter your GitHub username:${NC}"
read -p "GitHub Username: " username

if [ -z "$username" ]; then
    echo -e "${RED}❌ GitHub username is required!${NC}"
    exit 1
fi

# Construct repository URL
repo_url="https://github.com/$username/multi-omics-pharmacogenomics-platform.git"

echo ""
echo -e "${GREEN}🔧 Setting up repository...${NC}"
echo -e "${BLUE}Repository URL: $repo_url${NC}"

# Check if remote already exists
existing_remote=$(git remote get-url origin 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Remote 'origin' already exists: $existing_remote${NC}"
    echo -e "${YELLOW}Do you want to update it? (y/N)${NC}"
    read -p "" update_remote
    
    if [[ "$update_remote" == "y" || "$update_remote" == "Y" ]]; then
        echo -e "${GREEN}🔄 Updating remote origin...${NC}"
        git remote set-url origin "$repo_url"
    else
        echo -e "${RED}❌ Cancelled.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}➕ Adding remote origin...${NC}"
    git remote add origin "$repo_url"
fi

# Verify remote was added
remote_check=$(git remote get-url origin 2>/dev/null)
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to add remote repository!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Remote added successfully: $remote_check${NC}"

# Set main branch
echo -e "${GREEN}🌿 Setting main branch...${NC}"
git branch -M main

# Push to GitHub
echo -e "${GREEN}🚀 Pushing to GitHub...${NC}"
echo -e "${BLUE}This will upload all your code to: $repo_url${NC}"
echo ""

if git push -u origin main; then
    echo -e "${GREEN}🎉 SUCCESS! Repository pushed to GitHub!${NC}"
    echo -e "${BLUE}🌐 Your repository is now available at:${NC}"
    echo -e "${BLUE}   https://github.com/$username/multi-omics-pharmacogenomics-platform${NC}"
    echo ""
    echo -e "${GREEN}📊 Repository Statistics:${NC}"
    file_count=$(git ls-files | wc -l)
    commit_count=$(git rev-list --count HEAD)
    echo -e "${BLUE}   Files: $file_count${NC}"
    echo -e "${BLUE}   Commits: $commit_count${NC}"
    echo ""
    echo -e "${YELLOW}🔗 Next Steps:${NC}"
    echo -e "${BLUE}   1. Visit your repository on GitHub${NC}"
    echo -e "${BLUE}   2. Add a description and topics${NC}"
    echo -e "${BLUE}   3. Consider adding a LICENSE file${NC}"
    echo -e "${BLUE}   4. Set up GitHub Pages for documentation${NC}"
else
    echo -e "${RED}❌ Failed to push to GitHub!${NC}"
    echo ""
    echo -e "${YELLOW}💡 Common solutions:${NC}"
    echo -e "${BLUE}   1. Make sure the repository exists on GitHub${NC}"
    echo -e "${BLUE}   2. Check your GitHub username is correct${NC}"
    echo -e "${BLUE}   3. Ensure you're logged in to Git (git config --global user.name)${NC}"
    echo -e "${BLUE}   4. You might need to authenticate with GitHub${NC}"
    exit 1
fi
