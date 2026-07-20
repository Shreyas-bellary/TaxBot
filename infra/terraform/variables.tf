variable "project_id" {
  description = "GCP project that hosts TaxBot."
  type        = string
  default     = "gen-lang-client-0055378858"
}

variable "region" {
  description = "Cloud Run region."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "taxbot"
}

variable "migration_job_name" {
  description = "Cloud Run migration job name."
  type        = string
  default     = "taxbot-migrate"
}

variable "container_image" {
  description = "Initial placeholder image. CI/CD replaces it with Docker Hub builds."
  type        = string
  # Public Cloud Run sample image (not your Artifact Registry). Deploy swaps this.
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "revision" {
  description = "Source revision label used during initial provisioning."
  type        = string
  default     = "terraform"
}

variable "qdrant_url" {
  description = "Existing Qdrant Cloud HTTPS endpoint."
  type        = string

  validation {
    condition     = can(regex("^https://", var.qdrant_url))
    error_message = "qdrant_url must use HTTPS."
  }
}

variable "qdrant_collection" {
  type    = string
  default = "taxbot_child_nodes"
}

variable "rate_limit_answers_per_day" {
  type    = number
  default = 3

  validation {
    condition     = var.rate_limit_answers_per_day >= 1
    error_message = "rate_limit_answers_per_day must be positive."
  }
}

variable "max_instances" {
  type    = number
  default = 3
}
