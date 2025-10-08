variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "zone" {
  type        = string
  description = "GCP zone"
  default     = "asia-south1-c"
}

variable "instance_name" {
  type        = string
  description = "Compute instance name"
  default     = "mediamtx-relay"
}

variable "machine_type" {
  type        = string
  description = "Compute machine type"
  default     = "e2-medium"
}

variable "path_regex" {
  type        = string
  description = "Regex for allowed publish paths (MediaMTX)"
  default     = "^s([0-9]|[1-5][0-9]|6[0-3])$"
}

