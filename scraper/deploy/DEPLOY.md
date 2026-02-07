# Scraper AWS Deployment Guide

Deploy the vBulletin forum scraper to AWS using CloudFormation and GitHub Actions.

**Estimated cost**: ~$5-10 total (t3.small spot + 50GB EBS + S3 storage)

---

## Prerequisites

### 1. AWS CLI Setup

```bash
# Install AWS CLI if needed: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

# Configure with credentials that can create IAM users
aws configure
```

### 2. Create S3 Bucket

Choose a globally unique bucket name:

```bash
# Create the bucket (name must start with vlm3-scraper- for IAM policy to work)
aws s3 mb s3://vlm3-scraper-YOUR-UNIQUE-ID --region us-east-1

# Verify
aws s3 ls | grep vlm3-scraper
```

### 3. Create IAM User for GitHub Actions

Create a dedicated IAM user with least-privilege permissions:

```bash
# Create the IAM user
aws iam create-user --user-name github-actions-scraper

# Create access keys - SAVE THE OUTPUT!
aws iam create-access-key --user-name github-actions-scraper
```

**Important**: Save the `AccessKeyId` and `SecretAccessKey` from the output - you'll need these for GitHub secrets.

Now create and attach the IAM policy:

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
        "cloudformation:GetTemplate",
        "cloudformation:CreateChangeSet",
        "cloudformation:ExecuteChangeSet",
        "cloudformation:DescribeChangeSet",
        "cloudformation:DeleteChangeSet",
        "cloudformation:GetTemplateSummary"
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
      "Sid": "IAM",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:GetRole",
        "iam:GetRolePolicy",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:DeleteInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:GetInstanceProfile",
        "iam:PassRole",
        "iam:TagRole"
      ],
      "Resource": [
        "arn:aws:iam::*:role/vlm3-scraper-*",
        "arn:aws:iam::*:instance-profile/vlm3-scraper-*"
      ]
    },
    {
      "Sid": "IAMServiceLinkedRole",
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "arn:aws:iam::*:role/aws-service-role/spot.amazonaws.com/*",
      "Condition": {
        "StringEquals": {
          "iam:AWSServiceName": "spot.amazonaws.com"
        }
      }
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
        "ssm:GetParameter",
        "ssm:GetParameters"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:DeleteLogGroup",
        "logs:DescribeLogGroups",
        "logs:PutRetentionPolicy",
        "logs:TagResource"
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

# Get your AWS account ID and attach the policy
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws iam attach-user-policy \
  --user-name github-actions-scraper \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy

# Verify
aws iam get-user --user-name github-actions-scraper
aws iam list-attached-user-policies --user-name github-actions-scraper
```

---

## Deployment

### Step 1: Configure GitHub Repository Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions → Repository secrets**

Add these secrets:

| Secret Name | Required | Value |
|-------------|----------|-------|
| `AWS_ACCESS_KEY_ID` | Yes | AccessKeyId from IAM user creation |
| `AWS_SECRET_ACCESS_KEY` | Yes | SecretAccessKey from IAM user creation |
| `AWS_REGION` | No | `us-east-1` (default if not set) |
| `S3_BUCKET` | Yes | Your S3 bucket name (e.g., `vlm3-scraper-results`) |
| `E30M3_FORUM_URL` | Yes | Base URL of the forum to scrape |
| `OXYLABS_USERNAME` | No | Oxylabs proxy username (for Cloudflare bypass) |
| `OXYLABS_PASSWORD` | No | Oxylabs proxy password |

### Step 2: Deploy Infrastructure

1. Go to: **GitHub repo → Actions → "Forum Scraper"**
2. Click **"Run workflow"**
3. Select:
   - **Action**: `deploy`
   - **Instance type**: `t3.small` (default)
4. Click **"Run workflow"**

This creates:
- EC2 Spot instance (Amazon Linux 2023)
- IAM role for the instance
- Security group (HTTPS/HTTP egress + Oxylabs port 7777)
- CloudWatch log group
- Automatic S3 sync (every 6 hours)
- `.env` file with forum URL and proxy credentials from secrets

The deploy action is fully automated - no manual SSH/SSM required.

### Step 3: Validate with Test Action

After deploy succeeds, run the test action to validate everything works:

1. Go to Actions → "Forum Scraper"
2. Run workflow:
   - **Action**: `test`
3. This runs discovery and shows results in the workflow summary

### Step 4: Run the Full Scraper

**Via GitHub Actions (recommended):**
1. Go to Actions → "Forum Scraper"
2. Run workflow:
   - **Action**: `start`
   - **Stage**: `all` (or specific stage)
3. Monitor with `status` action
4. Save results with `sync` action

**Manual SSM access (optional, for debugging):**
```bash
# Install SSM plugin if needed (macOS)
brew install --cask session-manager-plugin

# Get instance ID
INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name e30-forum-scraper \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
  --output text)

# Connect via SSM
aws ssm start-session --target $INSTANCE_ID

# On instance:
sudo su - ec2-user
cd scraper
source .venv/bin/activate
tail -f data_src/forum/logs/scraper.log
```

---

## Scraper Stages

| Stage | Script | Purpose |
|-------|--------|---------|
| `discover` | `01_discover_forums.py` | Find all forums/subforums |
| `threads` | `02_scrape_threads.py` | Get thread listings |
| `posts` | `03_scrape_posts.py` | Scrape post content |
| `images` | `04_download_images.py` | Download attached images |
| `all` | `run_test_scrape.py` | Run all stages sequentially |

---

## Management

### Check Status
```bash
# Via GitHub Actions: action=status
# Or CLI:
INSTANCE_ID=$(aws cloudformation describe-stacks --stack-name e30-forum-scraper \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
aws ssm start-session --target $INSTANCE_ID
# Then: tail -f /home/ec2-user/scraper/data_src/forum/logs/scraper.log
```

### Sync to S3
```bash
# Via GitHub Actions: action=sync
# Automatic sync runs every 6 hours via cron
```

### Stop Scraper
```bash
# Via GitHub Actions: action=stop
# Sends SIGINT for graceful shutdown, then syncs to S3
```

### Teardown (delete everything)
```bash
# Via GitHub Actions: action=teardown
# Syncs final results to S3, then deletes the CloudFormation stack
```

---

## File Locations

**On EC2:**
```
/home/ec2-user/scraper/
├── data_src/forum/
│   ├── data/           # Scraped data (JSON)
│   ├── raw/            # Raw HTML
│   ├── checkpoints/    # Resume state
│   └── logs/           # scraper.log
└── .env                # Forum URL config
```

**On S3:**
```
s3://YOUR-BUCKET/
├── data_src/forum/     # Synced data
└── logs/               # Archived logs
```

---

## Troubleshooting

### Deployment Fails

Check CloudFormation events:
```bash
aws cloudformation describe-stack-events --stack-name e30-forum-scraper \
  --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
  --output table
```

### Delete Failed Stack

If stack is in ROLLBACK_COMPLETE state, delete it before retrying:
```bash
aws cloudformation delete-stack --stack-name e30-forum-scraper
aws cloudformation wait stack-delete-complete --stack-name e30-forum-scraper
```

### IAM Permission Errors

If you see "not authorized to perform" errors, update the IAM policy:
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
# Edit /tmp/scraper-policy.json with the missing permission, then:
aws iam create-policy-version \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy \
  --policy-document file:///tmp/scraper-policy.json \
  --set-as-default
```

**Note**: AWS limits you to 5 policy versions. Delete old versions if needed:
```bash
aws iam list-policy-versions --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy
aws iam delete-policy-version --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy --version-id v1
```

---

## Cleanup

Remove all AWS resources when done:

```bash
# 1. Teardown via GitHub Actions (recommended - syncs data first)
# Or manually:
aws cloudformation delete-stack --stack-name e30-forum-scraper
aws cloudformation wait stack-delete-complete --stack-name e30-forum-scraper

# 2. Delete IAM user (optional)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws iam detach-user-policy --user-name github-actions-scraper \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy
# Delete all policy versions except default, then delete policy
aws iam list-policy-versions --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy
aws iam delete-policy --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/github-actions-scraper-policy
# List and delete access keys
aws iam list-access-keys --user-name github-actions-scraper
aws iam delete-access-key --user-name github-actions-scraper --access-key-id <KEY_ID>
aws iam delete-user --user-name github-actions-scraper

# 3. Delete S3 bucket (optional - if you want to delete data)
aws s3 rb s3://YOUR-BUCKET --force
```

---

## Notes

- **Spot interruption**: Uses persistent spot with stop behavior - resumes when capacity available
- **Checkpointing**: Scraper saves progress every 10 items, auto-resumes on restart
- **Rate limiting**: 1.5-2.5s delay between requests (configurable in `scraper_config.yaml`)
- **VPC**: Auto-detects default VPC during deployment
- **First-time Spot users**: The IAM policy includes permission to create the EC2 Spot service-linked role
