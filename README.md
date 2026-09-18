# Fall Detection AI 🚨

A real-time fall detection system built with **OpenCV** and **MediaPipe** pose detection, with automatic alerts via **AWS SNS**.

Built for the OpenCV AI Competition 2026, powered by AWS.

## How it works

1. The webcam feed is processed with MediaPipe's pose detection to track body landmarks.
2. The system calculates the angle of the person's torso in real time.
3. If the torso angle exceeds **45°** for more than **1.5 seconds**, a fall is confirmed.
4. An automatic **email alert** is sent via AWS SNS to notify a caregiver or family member.

## Tech stack

- Python 3.11
- OpenCV
- MediaPipe (pose detection)
- AWS SNS (email notifications)
- boto3 (AWS SDK for Python)

## Setup

1. Install dependencies:
