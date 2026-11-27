# Model card: EdgeAuth face verification

Format follows Mitchell et al., *Model Cards for Model Reporting* (2019). Numbers come from
[`reports/results/lfw_verification.json`](reports/results/lfw_verification.json); see
[`reports/findings.md`](reports/findings.md) for the analysis.

## Model details

| | |
|---|---|
| Task | 1:1 face verification: does this capture match the claimed person's template? |
| Detector | YuNet (2023mar), OpenCV Zoo, ONNX |
| Recogniser | SFace (2021dec), 128-d embedding, cosine similarity; FP32 and INT8 (OpenCV Zoo quantisation) |
| Baseline | dlib ResNet-34 128-d embedding via `face_recognition` 1.3 (the 2022 smart lock), Euclidean distance |
| Presentation attack screen | OpenVINO `anti-spoof-mn3` (lab pipeline only; not part of the LFW study) |
| Weights | Pre-trained third-party weights, pinned by SHA-256; no training or fine-tuning in this repo |
| Licences | SFace: Apache-2.0, YuNet: MIT (OpenCV Zoo); dlib models: CC0 (`face_recognition_models`); LFW: research use |

## Intended use

* **Intended:** studying the accuracy, fairness and edge-hardware cost of face verification;
  demonstrating calibrated, cost-aware threshold selection; teaching.
* **Not intended:** physical access control, law enforcement, surveillance, 1:N identification
  of people who have not enrolled, or any decision with legal or similarly significant effect.
  A deployment would also need a trusted capture path, liveness testing to ISO/IEC 30107-3,
  a second factor and a fallback route (see NIST SP 800-63B-4 §3.2.3).

## Evaluation data

**LFW**, official View 2 protocol: 6,000 pairs (3,000 genuine, 3,000 impostor) in 10
identity-disjoint folds. Thresholds are chosen on folds 1–5 and tested on 6–10, and vice
versa, so every reported decision is out-of-sample.

LFW is a *web-photo* benchmark of mostly public figures. It is easier than a doorway
camera in some ways (photographers, good lighting) and harder in others (pose, age gaps
between photos). It is also demographically skewed (mostly male, mostly lighter-skinned), so
minority-group estimates have wide intervals.

**Subgroup labels** come from the Kumar et al. (2009) LFW attribute classifiers. They are
*estimated* attributes, not self-identification. Ethnicity is the argmax of four
classifier scores (Asian / White / Black / Indian), which is a coarse, imperfect proxy.
Pose is estimated from YuNet landmarks.

## Metrics

Attempt-level **FAR** (stranger accepted) and **FRR** (genuine user rejected), where a
failure to detect a face counts as a rejection. Also EER, ROC AUC and failure-to-acquire
rate. Intervals are 95% percentile bootstrap intervals over pairs.

Three threshold policies are compared: EER point, FAR ≤ 1%, and a **cost-based** policy
that minimises `10·FAR + 1·FRR` (a stranger getting in is taken as ten times worse than the
owner having to retry). Subgroup results use the cost-based policy.

## Results

See [`reports/findings.md`](reports/findings.md) for headline numbers, DET curves,
subgroup forest plots and the edge benchmark.

## Ethical considerations and caveats

* Face embeddings are biometric special-category data under UK GDPR Art. 9. The lab stores
  templates encrypted (AES-GCM), keyed by pseudonymous IDs, and the study uses only a public
  benchmark. No face images are committed; figures show aggregates only.
* Bootstrap intervals resample pairs, not people. LFW pairs share identities, so the true
  uncertainty is somewhat larger than shown.
* Results on LFW do not transfer to a specific door camera without re-calibrating on
  consented data from that camera.
* The pipeline has not been tested for spoof resistance on the LFW study, by design: LFW
  images are photographs and would be (correctly) rejected by the PAD screen.
