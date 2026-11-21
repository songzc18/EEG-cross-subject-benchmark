# EEG Foundation Models: Generalization and Adaptation

This repository contains research code for evaluating EEG decoding models across subjects and recording sessions. The project examines individual differences in model performance and compares different training and adaptation strategies.

## Research Questions

1. How does decoding performance vary across individuals under different training protocols?
2. Can a model trained on other subjects be adapted to a new subject using limited individual data?
3. Are differences in model performance and adaptation benefits stable across recording sessions?

## Evaluation Protocols

| Protocol | Description |
| --- | --- |
| POP | Population training, including target-subject data |
| LOSO | Leave-one-subject-out training |
| CAL | Population model excluding the target subject, followed by individual calibration |
| SELF | Training using only target-subject data |
