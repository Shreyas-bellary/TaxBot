# TaxBot

TaxBot helps you understand US taxes by answering your questions using official IRS documents — forms, instructions, and publications from irs.gov.
Unlike a general chatbot, TaxBot does not make things up. Every answer is based on retrieved IRS text, and you can click through to the original source to verify it yourself.

**Example:** *"Can I deduct home office expenses?"* → a clear answer with links to the relevant IRS guidance.

## Local development

Requirements: Python 3.13, Poetry 2.4, Node 22, and access to the external
Postgres, Qdrant, Hugging Face, and Gemini services.

```bash
cp .env.example .env
make install
make migrate
make api
```

Run the frontend separately:

```bash
make frontend-install
make frontend-dev
```

## Container

The production image builds the React application and serves it from FastAPI.
It uses CPU-only PyTorch, includes the fine-tuned reranker base model, runs as a
non-root user, and listens on port 8080.

```bash
make docker-build
make docker-run
curl http://localhost:8080/healthz
```

The image never includes `.env`. Local Docker uses `--env-file .env`; Cloud Run
uses Secret Manager.

## First GCP deployment

The target is project `gen-lang-client-0055378858` in `us-central1`. Cloud Run provides
the public HTTPS `run.app` URL.

### 1. Bootstrap GCP

Install `gcloud` and Terraform 1.7+, enable billing on the project, then:

```bash
gcloud auth application-default login
gcloud config set project gen-lang-client-0055378858

cd infra/bootstrap
terraform init
terraform apply
cd ../..
```

Bootstrap creates required APIs, Secret Manager containers, the Terraform state
bucket, and GitHub OIDC service accounts. Container images are published to a
**public Docker Hub** repository. Keep the local
`infra/bootstrap/terraform.tfstate` in a secure backup; it contains resource
metadata but no secret payloads.

### 2. Upload secret values

Populate `.env` locally, then upload only the selected values:

```bash
./infra/scripts/sync-secrets.sh .env
```

Run the script again whenever a secret rotates. Terraform never receives secret
values, so they do not enter state.

### 3. Provision Cloud Run

```bash
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
# Set qdrant_url in terraform.tfvars to the existing Qdrant Cloud HTTPS URL.

terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform apply
```

The first apply creates a placeholder revision plus the migration job. The next
successful push to `main` replaces it with the TaxBot image.

### 4. Configure GitHub repository variables

Read the bootstrap outputs:

```bash
terraform -chdir=infra/bootstrap output github_repository_variables
```

Create these repository variables in `Shreyas-bellary/TaxBot`:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_DEPLOY_SERVICE_ACCOUNT`
- `GCP_TERRAFORM_SERVICE_ACCOUNT`
- `TAXBOT_QDRANT_URL` (the existing Qdrant Cloud HTTPS URL)
- `DOCKERHUB_USERNAME` (Docker Hub account that owns the public `taxbot` repo)

Also create a repository **secret**:

- `DOCKERHUB_TOKEN` — a Docker Hub access token with push access to
  `DOCKERHUB_USERNAME/taxbot` (create a public repository named `taxbot` first)

The workflows use keyless GitHub OIDC for GCP. Do not create or upload a GCP
service account key. The `production` GitHub environment may optionally require
manual approval.

## CI/CD

Pull requests and pushes to `main` run:

- Ruff, mypy, migrations, and pytest against Postgres 16
- frontend ESLint and production build
- Terraform formatting and validation
- a full production Docker build

After CI succeeds on a `main` push, deployment:

1. reconciles Terraform infrastructure;
2. builds and pushes an immutable commit-SHA image to public Docker Hub;
3. runs the Cloud Run migration job and stops on failure;
4. deploys the same image to Cloud Run;
5. smoke-tests `/healthz` and the frontend.

Only one production deployment runs at a time, and newer pushes cancel stale
deployments.

## Operations

Get the public URL:

```bash
gcloud run services describe taxbot \
  --project=gen-lang-client-0055378858 \
  --region=us-central1 \
  --format='value(status.url)'
```

List revisions and roll traffic back:

```bash
gcloud run revisions list \
  --service=taxbot \
  --project=gen-lang-client-0055378858 \
  --region=us-central1

gcloud run services update-traffic taxbot \
  --project=gen-lang-client-0055378858 \
  --region=us-central1 \
  --to-revisions=REVISION_NAME=100
```

Cloud Run scales from zero to three instances. Each instance has 1 CPU, 2 GiB
memory, and request concurrency 10.