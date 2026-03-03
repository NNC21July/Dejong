# Firewatch: AI-Assisted Fire Risk Validation and Escalation

Firewatch is a fire incident validation pipeline that combines computer vision, risk scoring, and operational response logic.

The project is designed to reduce false alarms while still escalating quickly when high-risk events are detected.

## What this project does

Firewatch ingests CCTV footage, analyzes it in short intervals, and classifies scene risk into operational levels.  
It supports:

- fire/smoke visual detection
- risk scoring with weighted signals
- optional contextual reasoning for difficult scenes
- alerting and escalation workflows
- authority notification payload generation
- emergency public announcement triggers

## Repository structure

- `model_training/`
  - Training and fine-tuning workflows for the detection model.
  - Model classes include `controlled_fire`, `fire`, and `smoke`.
- `classification/`
  - Runtime analysis pipeline for interval-based video classification.
  - Produces per-interval artifacts (JSON metrics and JPG frame evidence).
- `firewatch/` and `firewatch_project/`
  - Django dashboard, APIs, monitoring loop, escalation logic, and notification integrations.
- `testbench/`
  - Step-by-step setup and testing guide for new users.

## Working principles

### 1. CCTV feed analysis

The system processes footage in time windows (for example, 5-second intervals) and runs model inference on sampled frames.

### 2. Fire validation with weighted scoring

Detection outputs are aggregated into risk signals (for example fire confidence, spread/flicker behavior, smoke presence, controlled vs uncontrolled fire evidence).  
A weighted scoring model converts these signals into a final risk score.

### 3. Optional contextual layer for challenging scenes

When the scene is uncertain or locally appears severe, Firewatch can invoke an OpenAI contextual step to improve decision reliability.  
If no valid API key is available, the pipeline continues in local-only mode.

### 4. Classification levels

Firewatch maps scores into operational levels:

- `No Fire Risk`
- `Elevated Risk`
- `Hazard`
- `Emergency`

Each level maps to different response behavior.

### 5. Response and escalation

At higher severities, Firewatch triggers response workflows such as:

- dashboard hazard/emergency alerts
- escalation actions for authorities
- Telegram escalation payload delivery
- emergency evacuation/public announcement events

## Why this exists

Traditional fire alerts can be noisy in real environments. Firewatch focuses on:

- validating alarms before escalation when possible
- escalating quickly when evidence indicates real danger
- preserving explainable evidence for each decision window

## Setup and test guide

Detailed installation, environment setup, and end-to-end test instructions are in:

- `testbench/README.md`

Use that document as the canonical onboarding guide for running and validating this project.
