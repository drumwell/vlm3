# Scraper AWS Deployment Guide

Deploy the vBulletin forum scraper to AWS using CloudFormation and GitHub Actions.

**Estimated cost**: ~$5-10 total (t3.small spot + 50GB EBS + S3 storage)

---

## Prerequisites

### 1. AWS CLI Setup
- AWS CLI installed locally
- Configured with credentials that can create IAM users: `aws configure`

### 2. Create Dedicated IAM User for GitHub Actions

**Why**: Don't use root or personal admin credentials in CI/CD. Create a dedicated user with only the permissions needed.

```bash
# Create the IAM user
aws iam create-user --user-name github-actions-scraper

# Create access keys (save these - you'll need them for GitHub secrets)
aws iam create-access-key --user-name github-actions-scraper
```

**Save the output** - you'll get `AccessKeyId` and `SecretAccessKey`. These go into GitHub secrets.

Now attach the required permissions policy:

```bash
# Create the policy document
cat > /tmp/scraper-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudFormation",
      "Effect": "Allow",
      "Action": [
        "cloudformation:CreateStack",
        "cloudformation:UpdateStack",
        "cloudformation:DeleteStack",
        "cloudformation:DescribeStacks",
        "cloudformation:DescribeStackEvents",
        "cloudformation:GetTemplate"
      ],
      "Resource": "arn:aws:cloudformation:*:*:stack/e30-forum-scraper/*"
    },
    {
      "Sid": "EC2",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:CreateTags",
        "ec2:CreateSecurityGroup",
        "ec2:DeleteSecurityGroup",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:DescribeSecurityGroups",
        "ec2:CreateLaunchTemplate",
        "ec2:DeleteLaunchTemplate",
        "ec2:DescribeLaunchTemplates",
        "ec2:DescribeLaunchTemplateVersions",
        "ec2:DescribeImages",
        "ec2:DescribeSubnets",
        "ec2:DescribeVpcs"
      ],
      "Resource": "*"
    },
    {
      "Sid": "IAMPassRole",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:GetRole",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:DeleteInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:GetInstanceProfile",
        "iam:PassRole"
      ],
      "Resource": [
        "arn:aws:iam::*:role/vlm3-scraper-*",
        "arn:aws:iam::*:instance-profile/vlm3-scraper-*"
      ]
    },
    {
      "Sid": "S3",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::vlm3-scraper-*",
        "arn:aws:s3:::vlm3-scraper-*/*"
      ]
    },
    {
      "Sid": "SSM",
      "Effect": "Allow",
      "Action": [
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:DescribeInstanceInformation",
        "ssm:GetParameter"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:DeleteLogGroup",
        "logs:DescribeLogGroups"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/scraper/*"
    }
  ]
}
EOF

# Create the policy
aws iam create-policy \
  --policy-name github-actions-scraper-policy \
  --policy-document file:///tmp/scraper-policy.json

# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Attach policy to user
aws iam attach-user-policy \
  --user-name github-actions-scraper \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy
```

**Verify the user was created:**
```bash
aws iam get-user --user-name github-actions-scraper
aws iam list-attached-user-policies --user-name github-actions-scraper
```

### 3. Create S3 Bucket

Choose a globally unique bucket name (e.g., `vlm3-scraper-results-<your-id>`):

```bash
# Create the bucket
aws s3 mb s3://vlm3-scraper-results --region us-east-1

# Verify it was created
aws s3 ls | grep vlm3-scraper
```

**Note**: Bucket names must be globally unique. If the name is taken, add a suffix like `-123` or your initials.

### 4. Forum URL
- Have the target forum URL ready for configuration

---

## Deployment Steps

### Step 1: Configure GitHub Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions**

Add these **required** secrets:

| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | AccessKeyId from step 2 |
| `AWS_SECRET_ACCESS_KEY` | SecretAccessKey from step 2 |
| `AWS_REGION` | `us-east-1` (or your preferred region) |
| `S3_BUCKET` | Your S3 bucket name (without `s3://`) |

### Step 2: Deploy Infrastructure

1. Go to: **GitHub repo → Actions → "Forum Scraper" workflow**
2. Click **"Run workflow"**
3. Select:
   - **Action**: `deploy`
   - **Instance type**: `t3.small` (default)
4. Click **"Run workflow"**

This deploys:
- EC2 Spot instance (Amazon Linux 2023)
- IAM role with S3/SSM/CloudWatch permissions
- Security group (HTTPS/HTTP egress only)
- CloudWatch log group
- Automatic S3 sync cron (every 6 hours)

### Step 3: Configure Forum URL on Instance

After deployment completes, connect via SSM:
```bash
# Get instance ID from CloudFormation outputs
INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name e30-forum-scraper \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
  --output text)

# Connect to instance
aws ssm start-session --target $INSTANCE_ID
```

On the instance, set the forum URL:
```bash
cd /home/ec2-user/scraper
echo "E30M3_FORUM_URL=https://your-forum-url.com" > .env
```

### Step 4: Start the Scraper

**Option A: Via GitHub Actions**
1. Go to Actions → "Forum Scraper"
2. Run workflow with:
   - **Action**: `start`
   - **Stage**: `discover` (first run) or `all` (full scrape)

**Option B: Via SSM (manual)**
```bash
aws ssm start-session --target $INSTANCE_ID
# Then on instance:
cd /home/ec2-user/scraper
source .venv/bin/activate
python scraper/01_discover_forums.py  # First: discover forums
python scraper/run_test_scrape.py --stage all --forum-id <ID>  # Full scrape
```

---

## Scraper Stages

| Stage | Script | Purpose |
|-------|--------|---------|
| `discover` | `01_discover_forums.py` | Find all forums/subforums |
| `threads` | `02_scrape_threads.py` | Get thread listings |
| `posts` | `03_scrape_posts.py` | Scrape post content |
| `images` | `04_download_images.py` | Download attached images |
| `all` | `run_test_scrape.py` | Run all stages |

---

## Management Commands

### Check Status
```bash
# Via GitHub Actions: Run workflow with action=status
# Or via CLI:
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["pgrep -a python", "tail -20 /home/ec2-user/scraper/forum_archive/logs/scraper.log"]'
```

### Manual S3 Sync
```bash
# Via GitHub Actions: Run workflow with action=sync
```

### Stop Scraper (graceful)
```bash
# Via GitHub Actions: Run workflow with action=stop
# Sends SIGINT, waits, then syncs to S3
```

### Teardown (delete everything)
```bash
# Via GitHub Actions: Run workflow with action=teardown
# Syncs final results to S3, then deletes stack
```

---

## File Locations

**On EC2 instance:**
- Code: `/home/ec2-user/scraper/`
- Output: `/home/ec2-user/scraper/forum_archive/`
- Logs: `/home/ec2-user/scraper/forum_archive/logs/scraper.log`
- Checkpoints: `/home/ec2-user/scraper/forum_archive/checkpoints/`

**On S3:**
- Results: `s3://YOUR-BUCKET/forum_archive/`
- Logs: `s3://YOUR-BUCKET/logs/`

---

## Verification Checklist

After deployment, verify:
- [ ] CloudFormation stack shows CREATE_COMPLETE
- [ ] Can connect via SSM: `aws ssm start-session --target $INSTANCE_ID`
- [ ] Python environment works: `source .venv/bin/activate && python --version`
- [ ] Forum URL is set: `cat .env`
- [ ] Discover stage completes: `python scraper/01_discover_forums.py`

---

## Notes

- **Spot interruption**: Instance uses persistent spot with stop behavior - will resume when capacity available
- **Checkpointing**: Scraper saves progress every 10 items, auto-resumes on restart
- **Rate limiting**: 1.5-2.5s delay between requests (configurable in `scraper_config.yaml`)
- **S3 sync**: Automatic every 6 hours via cron, or manual via GitHub Actions

---

## Cleanup

To remove all AWS resources when done:

```bash
# Via GitHub Actions
# Run workflow with action=teardown

# Or manually:
aws cloudformation delete-stack --stack-name e30-forum-scraper
aws cloudformation wait stack-delete-complete --stack-name e30-forum-scraper

# Optionally delete the IAM user (if no longer needed)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws iam detach-user-policy --user-name github-actions-scraper \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy
aws iam delete-policy --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy
aws iam delete-access-key --user-name github-actions-scraper --access-key-id <KEY_ID>
aws iam delete-user --user-name github-actions-scraper

# S3 bucket (if you want to delete data too)
aws s3 rb s3://YOUR-BUCKET --force
```
