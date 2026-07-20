variable "project_id" {
  description = "GCP project that hosts TaxBot."
  type        = string
  default     = "gen-lang-client-0055378858"
}

variable "region" {
  description = "GCP region for Cloud Run and the Terraform state bucket."
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  description = "GitHub repository allowed to exchange OIDC tokens."
  type        = string
  default     = "Shreyas-bellary/TaxBot"
}
