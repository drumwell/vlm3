# AWS Deployment for Forum Scraper

Deploy and run the forum scraper on AWS using GitHub Actions.

> **Note:** The workflow file lives at `.github/workflows/scraper.yml` (repo root, required by GitHub)
> but references templates from `scraper/deploy/`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GitHub Actions                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Deploy    │  │    Start    │  │    Sync     │          │
│  │   Stack     │  │   Scraper   │  │   Results   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                         AWS                                  │
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │   CloudFormation │      │       S3         │             │
│  │   (Infra as Code)│      │  (Result Store)  │             │
│  └────────┬─────────┘      └────────▲─────────┘             │
│           │                         │                        │
│           ▼                         │                        │
│  ┌──────────────────┐               │                        │
│  │   EC2 Spot       │───────────────┘                        │
│  │   Instance       │  (sync every 6h)                       │
│  │                  │                                        │
│  │  - Python 3      │                                        │
│  │  - Scraper code  │                                        │
│  │  - forum_archive/│                                        │
│  └──────────────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │   SSM Agent      │  (secure remote access)                │
│  └──────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Create S3 Bucket

```bash
aws s3 mb s3://your-scraper-bucket-name --region us-east-1
```

### 2. Configure GitHub Secrets

Go to your repository Settings → Secrets and variables → Actions, and add:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key with EC2/S3/SSM/CloudFormation permissions |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `AWS_REGION` | AWS region (e.g., `us-east-1`) |
| `S3_BUCKET` | S3 bucket name for results |

### 3. Required IAM Permissions

The AWS user/role needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudformation:*",
        "ec2:*",
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:DeleteInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:PassRole",
        "iam:GetRole",
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:StartSession",
        "s3:*",
        "logs:*"
      ],
      "Resource": "*"
    }
  ]
}
```

## Usage

### Deploy Infrastructure

1. Go to Actions → Forum Scraper → Run workflow
2. Select action: `deploy`
3. Choose instance type (t3.small recommended)
4. Click "Run workflow"

This creates:
- EC2 spot instance (cost-effective)
- IAM role with S3/SSM access
- Security group (outbound HTTPS only)
- CloudWatch log group

### Run Scraper Stages

**Stage 1: Discover Forums**
```
Action: start
Stage: discover
```

**Stage 2: Scrape Threads** (after discovering forum IDs)
```
Action: start
Stage: threads
Forum ID: <id from forums.json>
```

**Stage 3: Scrape Posts**
```
Action: start
Stage: posts
Forum ID: <id>
```

**Stage 4: Download Images**
```
Action: start
Stage: images
Forum ID: <id>
```

**Run All Stages** (once you have forum ID)
```
Action: start
Stage: all
Forum ID: <id>
```

### Monitor Progress

```
Action: status
```

Shows:
- Running Python processes
- Last 50 log lines
- Checkpoint files
- Disk usage

### Sync Results to S3

```
Action: sync
```

Manually sync results to S3. Also runs automatically every 6 hours.

### Stop Scraper

```
Action: stop
```

Gracefully stops scraper (sends SIGINT for clean checkpoint save), then syncs to S3.

### Teardown Infrastructure

```
Action: teardown
```

Syncs final results to S3, then deletes all AWS resources.

## Cost Estimates

Using t3.small spot instance in us-east-1:

| Component | Cost |
|-----------|------|
| EC2 Spot (t3.small) | ~$0.005/hour = ~$3.60/month |
| EBS (50GB gp3) | ~$4/month |
| S3 Storage | ~$0.023/GB/month |
| Data Transfer | First 100GB free, then $0.09/GB |

**For a full scrape (~250 hours):**
- EC2: ~$1.25
- EBS: ~$4 (if running for a month)
- S3: Depends on data size

**Total: ~$5-10** for the full scrape.

## Manual Access

Connect to the instance via SSM (no SSH key needed):

```bash
# Get instance ID
INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name e30-forum-scraper \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" \
  --output text)

# Start session
aws ssm start-session --target $INSTANCE_ID
```

## Recovering from Interruption

Spot instances can be interrupted. The scraper handles this gracefully:

1. Checkpoints are saved regularly
2. S3 sync runs every 6 hours automatically
3. On restart, run `deploy` again - it will resume from checkpoints

To restore checkpoints from S3:

```bash
# On instance
aws s3 sync s3://your-bucket/forum_archive /home/ec2-user/scraper/forum_archive
```

## Downloading Results

After scraping is complete:

```bash
# Download all results
aws s3 sync s3://your-bucket/forum_archive ./forum_archive

# Download just the data files
aws s3 sync s3://your-bucket/forum_archive/data ./forum_archive/data
```

## Troubleshooting

### Instance Not Starting
- Check CloudFormation events in AWS Console
- Verify IAM permissions
- Check spot instance capacity in region

### Scraper Not Running
- Check status via GitHub Actions
- Connect via SSM and check logs: `tail -f /home/ec2-user/scraper/forum_archive/logs/scraper.log`

### S3 Sync Failing
- Check IAM role has S3 permissions
- Verify bucket name in secrets
