provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_apis = toset([
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "serviceusage.googleapis.com",
    "sts.googleapis.com",
  ])

  secret_ids = toset([
    "taxbot-gemini-api-key",
    "taxbot-huggingface-api-token",
    "taxbot-openrouter-api-key",
    "taxbot-postgres-dsn",
    "taxbot-qdrant-api-key",
    "taxbot-unstructured-api-key",
  ])

  github_repository_attribute = "attribute.repository/${var.github_repository}"
}

resource "google_project_service" "required" {
  for_each = local.required_apis

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_storage_bucket" "terraform_state" {
  name                        = "${var.project_id}-terraform-state"
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 20
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret" "taxbot" {
  for_each = local.secret_ids

  project   = var.project_id
  secret_id = each.value

  replication {
    auto {}
  }

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"
  description               = "Keyless GitHub Actions identities for TaxBot"

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "TaxBot GitHub repository"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  attribute_condition = "assertion.repository == '${var.github_repository}' && assertion.ref == 'refs/heads/main'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "github_deployer" {
  project      = var.project_id
  account_id   = "taxbot-github-deployer"
  display_name = "TaxBot GitHub application deployer"
}

resource "google_service_account" "github_infra" {
  project      = var.project_id
  account_id   = "taxbot-github-infra"
  display_name = "TaxBot GitHub Terraform runner"
}

resource "google_service_account_iam_member" "github_deployer_wif" {
  service_account_id = google_service_account.github_deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/${local.github_repository_attribute}"
}

resource "google_service_account_iam_member" "github_infra_wif" {
  service_account_id = google_service_account.github_infra.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/${local.github_repository_attribute}"
}

resource "google_project_iam_member" "deployer_roles" {
  for_each = toset([
    "roles/run.developer",
    "roles/run.invoker",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_deployer.email}"
}

resource "google_project_iam_member" "infra_roles" {
  for_each = toset([
    "roles/iam.serviceAccountAdmin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/run.admin",
    "roles/secretmanager.admin",
    "roles/serviceusage.serviceUsageAdmin",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_infra.email}"
}

resource "google_storage_bucket_iam_member" "infra_state_admin" {
  bucket = google_storage_bucket.terraform_state.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.github_infra.email}"
}
