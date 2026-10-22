# PEWTAF: A Privacy-Aware Lightweight Edge Intelligent Framework for Real-Time Lethal Threat Interception and Risk Assessment

Official code repository for the research paper:

**"A Privacy-Aware Lightweight Edge Intelligent Framework for Real-Time Lethal Threat Interception and Risk Assessment"**

*Scientific Reports — Springer Nature*

## Overview

PEWTAF is a lightweight, edge-based framework designed for real-time weapon detection, temporal threat validation, and contextual risk assessment. It integrates visual perception and threat analysis on resource-constrained hardware to support privacy-aware surveillance without continuously transmitting raw video to external servers.

The framework combines a hybrid YOLOv10–MobileViTv3 architecture with spatial attention, temporal consistency analysis, and deterministic threat scoring. The system is designed for deployment on a Raspberry Pi 5 with a Hailo-8L AI accelerator.

## Key Components

- **Hybrid YOLOv10–MobileViTv3:** Combines local visual feature extraction with global contextual representation.
- **Multi-Scale Hybrid Spatial Attention (MSHSA):** Refines spatial features to improve weapon localization, particularly for small or partially occluded objects.
- **Temporal Threat Validation Module (TTVM):** Evaluates detections across consecutive frames to suppress transient false alarms.
- **Threat Assessment Network (TAN):** Assigns threat scores, risk levels, and alert priorities using detection confidence, threat severity, and temporal persistence.
- **Privacy-Aware Edge Processing:** Processes video locally and is designed to transmit compact threat metadata rather than raw surveillance footage.

## System Hardware

| Component | Specification |
|---|---|
| Edge platform | Raspberry Pi 5 (8 GB RAM) |
| AI accelerator | Hailo-8L (13 TOPS) |
| Detection architecture | YOLOv10–MobileViTv3 |
| Attention mechanism | MSHSA |
| Temporal validation | TTVM |
| Threat assessment | TAN |

## Reported Results

The proposed framework achieved the following results under the experimental configuration reported in the paper.

| Evaluation Metric | Reported Result |
|---|---:|
| Precision | 95.36% |
| Recall | 93.74% |
| F1-score | 94.54% |
| mAP@0.5 | 96.58% |
| mAP@0.5:0.95 | 70.42% |
| Inference latency | 31.4 ms |
| Throughput | 31.8 FPS |
| Average system power consumption | 9.1 W |
| False-alarm reduction | 60.9% |
| Localization-error reduction | 34.1% |
| Actionable alerts | 92.8% |
| Alert-prioritization accuracy | 94.1% |
| Critical-threat identification rate | 96.3% |

*Note: These are the results reported in the manuscript and should be interpreted in the context of the experimental setup and evaluation protocol described in the paper.*

## Repository Contents

This repository contains the code implementation of the proposed PEWTAF framework. 

## Code Availability

The source code has been linked to a permanent, DOI-assigned Zenodo record in accordance with the journal's code-availability requirements.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.23510213

## Privacy and Responsible Use

PEWTAF is intended as a research framework for automated visual detection and risk-prioritized alerting. Its outputs should support human review rather than be treated as conclusive evidence of malicious intent.

Real-world deployment requires appropriate privacy safeguards, authorization to process surveillance footage, evaluation under representative environmental conditions, and human oversight of security-related decisions.

## License

The repository and source code are licensed under the MIT license.
