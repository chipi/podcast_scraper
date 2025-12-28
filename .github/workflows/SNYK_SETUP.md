# Snyk Security Scanning Setup

This repository uses [Snyk](https://snyk.io/) for comprehensive security vulnerability scanning of dependencies and Docker images.

## Setup Instructions

### 1. Get Your Snyk Token

1. Sign in to your [Snyk account](https://app.snyk.io/)
2. Go to **Settings** → **General** → **Account Settings**
3. Copy your **API Token** (or create a new one)

### 2. Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `SNYK_TOKEN`
5. Value: Paste your Snyk API token
6. Click **Add secret**

### 3. Verify Setup

Once the token is added, the Snyk workflow will automatically run on:

- **Push to main**: Scans dependencies and Docker images
- **Pull requests**: Scans and monitors dependencies
- **Weekly schedule**: Monitors dependencies for new vulnerabilities
- **Manual trigger**: Use "Run workflow" in GitHub Actions

## What Gets Scanned

### Dependencies (`snyk-dependencies` job)

- Scans Python dependencies from `requirements.txt`
- Reports vulnerabilities with severity threshold: **high** and above
- Uploads results to GitHub Code Scanning

### Docker Images (`snyk-docker` job)

- Builds the Docker image
- Scans the image for vulnerabilities in base image and installed packages
- Reports vulnerabilities with severity threshold: **high** and above
- Uploads results to GitHub Code Scanning

### Monitoring (`snyk-monitor` job)

- Monitors dependencies for new vulnerabilities over time
- Runs on pull requests and scheduled runs
- Creates/updates project in Snyk dashboard

## Viewing Results

### GitHub Code Scanning

- Go to **Security** → **Code scanning alerts** in your repository
- View Snyk findings alongside CodeQL results

### Snyk Dashboard

- Visit [app.snyk.io](https://app.snyk.io/)
- View detailed vulnerability information
- Get remediation advice
- Track vulnerability trends over time

## Configuration

### Severity Threshold

Currently set to `--severity-threshold=high` to focus on high and critical vulnerabilities. To change:

Edit `.github/workflows/snyk.yml`:

```yaml
args: --severity-threshold=medium  # or low, high, critical
```

### Scan Frequency

The workflow runs:

- On every push/PR (for immediate feedback)
- Weekly on Mondays (for ongoing monitoring)

To change the schedule, edit the `cron` expression in `.github/workflows/snyk.yml`.

## Troubleshooting

### Workflow Fails with "SNYK_TOKEN not found"

- Ensure the secret is added in GitHub repository settings
- Check that the secret name is exactly `SNYK_TOKEN`

### No Results in GitHub Code Scanning

- Snyk results are uploaded as SARIF files
- Check the workflow logs for upload status
- Ensure `security-events: write` permission is set (already configured)

### Docker Scan Fails

- Ensure Docker image builds successfully
- Check disk space (workflow includes cleanup steps)
- Verify Dockerfile path is correct

## Additional Resources

- [Snyk Documentation](https://docs.snyk.io/)
- [Snyk GitHub Actions](https://github.com/snyk/actions)
- [GitHub Code Scanning](https://docs.github.com/en/code-security/code-scanning)
