Uni-ACS (Unified Automatic Clinical Scoring) is a model agnostic approach to automatically generating clinical scores from SHAP compatibale clinical Machine Learning (ML) models. As Uni-ACS uses the original ML model and its explanations as a base for clinical score construction, it retains global and local interpretations of the original model, while promising to retain a significant proportion of the original ML model's predictive performance.

## Background

Modern interpretable ML tools (e.g. LIME, SHAP and explainable boosting machines) have made significant progress in explaining "black box" ML models. However, these tools' outputs continue to be unfamiliar to most clinicians. Clinicians' preferred tool of patient risk stratification remains to be clinical scores. Plausible reasons for such a preference as follows:
- Clinical scores are concise, thus they are easy to remember.
- Clinical scores can be easily correlated to the clinical context.
- Clinical scores can be calculated at bedside without the assistance of a machine.
Therefore, Uni-ACS aims to overcome this clinical translation problem by translating ML models into clinical scores.

## Requirements

- SHAP
- PyGAM
- CSAPS
- Scikit-Learn
- Pandas
- Numpy

## Quick Guide

Coming up!
