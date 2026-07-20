output "service_url" {
  description = "Public Cloud Run URL."
  value       = google_cloud_run_v2_service.taxbot.uri
}

output "service_name" {
  value = google_cloud_run_v2_service.taxbot.name
}

output "migration_job_name" {
  value = google_cloud_run_v2_job.migration.name
}

output "runtime_service_account" {
  value = google_service_account.runtime.email
}

output "migration_service_account" {
  value = google_service_account.migration.email
}

output "docker_hub_repository" {
  description = "Public Docker Hub repository used by deploy (set DOCKERHUB_USERNAME in GitHub)."
  value       = "docker.io/<DOCKERHUB_USERNAME>/taxbot"
}
