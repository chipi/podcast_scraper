# Branch Creation Checklist

## ⚠️ Always Follow This Before Creating a New Branch

### Step 1: Check Current State

```bash
git status
```

**What to look for:**
- ❌ If you see "Changes not staged for commit" → You have uncommitted changes
- ❌ If you see "Untracked files" → You have new files
- ✅ If you see "nothing to commit, working tree clean" → You're good to go!

### Step 2: Handle Uncommitted Changes

**Option A: Commit to Current Branch** (if changes belong to current work)

```bash
git add .
git commit -m "your message"
```

**Option B: Stash for Later** (if you want to save but not commit)

```bash
git stash

# Later: git stash pop

```

**Option C: Discard Changes** (if not needed)

```bash
git checkout .

# Or for specific files:

git checkout -- path/to/file
```

## Step 3: Ensure You're on Main and Up to Date

```bash
git checkout main
git pull origin main
```

### Step 4: Create Your Branch (Clean State)

```bash
git checkout -b issue-XXX-description
```

## Quick One-Liner Check

Before creating any branch, run:

```bash
git status --porcelain
```

**Expected output for clean state:**

```python

**If you see any output, handle it first!**

## What Happens If You Don't Follow This?

- ❌ Uncommitted changes from previous work get included in your new branch
- ❌ Your commit will show more files than you actually changed
- ❌ PR will show confusing diffs with unrelated changes
- ❌ Harder to review and understand what actually changed

## Example: Clean Branch Creation

```bash

# 1. Check status

$ git status
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean  ✅

# 2. Pull latest

$ git pull origin main
Already up to date.

# 3. Create branch

$ git checkout -b issue-117-output-organization
Switched to a new branch 'issue-117-output-organization'

# 4. Verify clean state

$ git status
On branch issue-117-output-organization
nothing to commit, working tree clean  ✅

```python

## If You Already Made the Mistake

If you already created a branch with uncommitted changes:

1. **Check what's actually different from main:**
   ```bash

   git diff main...HEAD --name-only
   ```

2. **If too many files, consider:**
   - Create a new branch from main (clean)
   - Cherry-pick only your actual changes
   - Or reset and recommit only your changes

3. **For the current PR:**
   - The PR diff on GitHub will show correctly (only differences from main)
   - But the commit history will be confusing
   - Consider amending the commit to only include your changes
