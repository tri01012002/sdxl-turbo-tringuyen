# Vietnamese Text-to-Image Generation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/tri01012002/Vietnamese-Text-to-Image)](https://github.com/tri01012002/Vietnamese-Text-to-Image/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/tri01012002/Vietnamese-Text-to-Image)](https://github.com/tri01012002/Vietnamese-Text-to-Image/network/members)

## Overview

This project is a production-ready text-to-image generation system optimized for Vietnamese prompts, achieving <15s latency per image while ensuring cultural accuracy. It uses fine-tuned AI models and a web interface for concurrent users, deployed on Hugging Face Spaces. Ideal for creative applications, education, or cultural content generation.

## Features

- **Real-Time Generation**: Low-latency image creation from Vietnamese text prompts.
- **Cultural Accuracy**: Custom LoRA fine-tuning for Vietnamese-specific elements.
- **Prompt Enhancement**: Automatic refinement for better results.
- **Web UI**: Gradio-based interface for easy use and multi-user support.
- **Deployment**: Hosted on Hugging Face for accessibility.

## Tech Stack

- **Core**: Python with PyTorch, Diffusers, Transformers.
- **UI**: Gradio for frontend.
- **Optimization**: xFormers for efficiency.
- **Deployment**: Hugging Face Hub/Spaces.

## Installation

1. Clone repo:
```bash
git clone https://github.com/tri01012002/Vietnamese-Text-to-Image.git
cd Vietnamese-Text-to-Image
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run:
```bash
python app.py
```
