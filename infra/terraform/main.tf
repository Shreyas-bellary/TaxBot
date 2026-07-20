locals {
  runtime_secret_env = {
    TAXBOT_GEMINI_API_KEY        = "taxbot-gemini-api-key"
    TAXBOT_HUGGINGFACE_API_TOKEN = "taxbot-huggingface-api-token"
    TAXBOT_POSTGRES_DSN          = "taxbot-postgres-dsn"
    TAXBOT_QDRANT_API_KEY        = "taxbot-qdrant-api-key"
  }

  runtime_env = {
    TAXBOT_ANSWER_LLM_MODEL                  = "gemma-4-31b-it"
    TAXBOT_ANSWER_LLM_PROVIDER               = "gemini"
    TAXBOT_BM25_MODEL                        = "Qdrant/bm25"
    TAXBOT_CORS_ALLOW_ORIGINS                = ""
    TAXBOT_EMBEDDING_DIMENSION               = "1024"
    TAXBOT_EMBEDDING_MODEL                   = "BAAI/bge-large-en-v1.5"
    TAXBOT_HF_EMBED_CONCURRENCY              = "2"
    TAXBOT_LOG_JSON                          = "true"
    TAXBOT_LOG_LEVEL                         = "INFO"
    TAXBOT_POSTGRES_STATEMENT_CACHE_SIZE     = "0"
    TAXBOT_QDRANT_COLLECTION                 = var.qdrant_collection
    TAXBOT_QDRANT_TIMEOUT_SECONDS            = "120"
    TAXBOT_QDRANT_URL                        = var.qdrant_url
    TAXBOT_RATE_LIMIT_ANSWERS_PER_DAY        = tostring(var.rate_limit_answers_per_day)
    TAXBOT_RATE_LIMIT_ENABLED                = "true"
    TAXBOT_RATE_LIMIT_TRUST_FORWARDED_FOR    = "true"
    TAXBOT_RERANKER_ENABLED                  = "true"
    TAXBOT_RERANKER_MODEL_PATH               = "/app/scripts/finetuned_model"
    TAXBOT_RERANKER_TOP_K                    = "12"
    TAXBOT_RETRIEVAL_CONFIDENCE_GATE_ENABLED = "true"
    TAXBOT_RETRIEVAL_MIN_HYBRID_SCORE        = "0.35"
    TAXBOT_RETRIEVAL_RRF_K                   = "60"
    TAXBOT_RETRIEVAL_TOP_K_CHILDREN          = "24"
    TAXBOT_RETRIEVAL_TOP_K_PARENTS           = "6"
    TAXBOT_ROUTER_LLM_MODEL                  = "gemma-4-26b-a4b-it"
    TAXBOT_ROUTER_LLM_PROVIDER               = "gemini"
    TAXBOT_STATIC_DIR                        = "/app/static"
    TAXBOT_USER_QUERY_END_TAG                = "USER_QUERY_END_5f3c1e"
    TAXBOT_USER_QUERY_START_TAG              = "USER_QUERY_START_5f3c1e"
  }

  deployer_service_account = "taxbot-github-deployer@${var.project_id}.iam.gserviceaccount.com"
  infra_service_account    = "taxbot-github-infra@${var.project_id}.iam.gserviceaccount.com"
}

data "google_secret_manager_secret" "runtime" {
  for_each = toset(values(local.runtime_secret_env))

  project   = var.project_id
  secret_id = each.value
}

resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "taxbot-runtime"
  display_name = "TaxBot Cloud Run runtime"
}

resource "google_service_account" "migration" {
  project      = var.project_id
  account_id   = "taxbot-migration"
  display_name = "TaxBot schema migration job"
}

resource "google_secret_manager_secret_iam_member" "runtime" {
  for_each = data.google_secret_manager_secret.runtime

  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "migration" {
  project   = var.project_id
  secret_id = data.google_secret_manager_secret.runtime["taxbot-postgres-dsn"].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.migration.email}"
}

resource "google_service_account_iam_member" "deployer_runtime_user" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.deployer_service_account}"
}

resource "google_service_account_iam_member" "deployer_migration_user" {
  service_account_id = google_service_account.migration.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.deployer_service_account}"
}

resource "google_service_account_iam_member" "infra_runtime_user" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.infra_service_account}"
}

resource "google_service_account_iam_member" "infra_migration_user" {
  service_account_id = google_service_account.migration.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.infra_service_account}"
}

resource "google_cloud_run_v2_service" "taxbot" {
  project             = var.project_id
  name                = var.service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = true

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "300s"
    max_instance_request_concurrency = 10

    labels = {
      "managed-by" = "terraform"
      "revision"   = substr(var.revision, 0, 63)
    }

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      ports {
        name           = "http1"
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        initial_delay_seconds = 0
        timeout_seconds       = 10
        period_seconds        = 10
        failure_threshold     = 24

        http_get {
          path = "/readyz"
          port = 8080
        }
      }

      liveness_probe {
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 60
        failure_threshold     = 3

        http_get {
          path = "/readyz"
          port = 8080
        }
      }

      dynamic "env" {
        for_each = local.runtime_env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.runtime_secret_env
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.runtime[env.value].secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      template[0].labels,
    ]
  }

  depends_on = [google_secret_manager_secret_iam_member.runtime]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = google_cloud_run_v2_service.taxbot.location
  name     = google_cloud_run_v2_service.taxbot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_job" "migration" {
  project             = var.project_id
  name                = var.migration_job_name
  location            = var.region
  deletion_protection = true

  template {
    template {
      service_account = google_service_account.migration.email
      timeout         = "600s"
      max_retries     = 1

      containers {
        image   = var.container_image
        command = ["taxbot-migrate"]

        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }

        env {
          name = "TAXBOT_POSTGRES_DSN"
          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.runtime["taxbot-postgres-dsn"].secret_id
              version = "latest"
            }
          }
        }

        env {
          name  = "TAXBOT_LOG_JSON"
          value = "true"
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image]
  }

  depends_on = [google_secret_manager_secret_iam_member.migration]
}
