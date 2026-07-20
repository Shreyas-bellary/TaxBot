output "terraform_state_bucket" {
  description = "Remote state bucket for the runtime Terraform stack."
  value       = google_storage_bucket.terraform_state.name
}

output "workload_identity_provider" {
  description = "GitHub Actions workload identity provider resource name."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "github_deployer_service_account" {
  value = google_service_account.github_deployer.email
}

output "github_infra_service_account" {
  value = google_service_account.github_infra.email
}

output "github_repository_variables" {
  description = "Values to configure as GitHub Actions repository variables."
  value = {
    GCP_PROJECT_ID                 = var.project_id
    GCP_REGION                     = var.region
    GCP_WORKLOAD_IDENTITY_PROVIDER = google_iam_workload_identity_pool_provider.github.name
    GCP_DEPLOY_SERVICE_ACCOUNT     = google_service_account.github_deployer.email
    GCP_TERRAFORM_SERVICE_ACCOUNT  = google_service_account.github_infra.email
  }
}
