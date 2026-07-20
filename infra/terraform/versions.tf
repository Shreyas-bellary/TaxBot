terraform {
  required_version = ">= 1.7.0"

  backend "gcs" {
    bucket = "gen-lang-client-0055378858-terraform-state"
    prefix = "cloud-run/production"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}
