terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.1"
    }
  }
}

provider "google" {
  project = var.project_id
  zone    = var.zone
}

resource "random_password" "frps_token" {
  length  = 32
  special = false
}

resource "google_compute_instance" "mediamtx" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["mediamtx"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 20
      type  = "pd-balanced"
    }
    auto_delete = true
  }

  network_interface {
    # Use default VPC
    network = "default"
    access_config {}
  }

  metadata_startup_script = templatefile("${path.module}/scripts/startup.sh", {
    path_regex = var.path_regex
    frp_token  = random_password.frps_token.result
  })
}

resource "google_compute_firewall" "mediamtx_allow" {
  name    = "mediamtx-allow"
  network = "default"

  target_tags = ["mediamtx"]

  allow {
    protocol = "tcp"
    ports    = ["7000", "8554", "1935", "8888", "8889", "8890", "9997", "9500-9600"]
  }

  allow {
    protocol = "udp"
    ports    = ["8000-8001", "8189", "8890", "9500-9600"]
  }

  direction     = "INGRESS"
  source_ranges = ["0.0.0.0/0"]
  description   = "Allow RTSP/RTMP/HLS/WebRTC/API and RTP/RTCP/ICE/SRT for MediaMTX"
}

output "external_ip" {
  value       = google_compute_instance.mediamtx.network_interface[0].access_config[0].nat_ip
  description = "Public IP of the relay"
}

output "frps_token" {
  value       = random_password.frps_token.result
  description = "Token for frps/frpc authentication"
  sensitive   = true
}
