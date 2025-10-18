# Project Structure

```
d_final/
├── AGENTS.md
├── STANDARDS.md
├── STRUCTURE.md
├── config/
│   ├── config_infer_primary.txt   # Detector config (TrafficCamNet)
│   ├── frpc.ini                   # FRP client template
│   └── rtsp_smoketest.txt         # DeepStream source30 demo config (30 FPS)
├── models/
│   ├── labels.txt                 # TrafficCamNet labels
│   └── resnet18_trafficcamnet_pruned.onnx  # Detector ONNX (Git LFS)
└── relay/
    ├── main.tf                    # Terraform for the relay VM
    ├── scripts/startup.sh         # Writes MediaMTX + FRPS configs and launches containers
    ├── test.sh                    # Helper to probe relay ports
    └── variables.tf               # Terraform variables (zone, project, instance name)
```

Nothing else is needed for the live demo.
