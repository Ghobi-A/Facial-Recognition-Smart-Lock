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

| Dataset | People | Pairs | Purpose | Labels |
|---|---|---|---|---|
| LFW View 2 | 4,281 | 6,000 | primary benchmark | Kumar et al. (2009) attribute classifiers (estimated) |
| CFP-FF / CFP-FP | 500 | 5,272 / 5,258 | frontal baseline / 90° profile | profile flag |
| CelebA test split | 1,000 | 5,866 | replication of sex and condition effects | annotated attributes |
| RFW, BFW | – | – | ethnicity replication (loaders ready, data pending access) | dataset-assigned |

Each dataset is split into two identity-disjoint halves. Thresholds chosen on one half are
tested on the other, so every reported decision is out-of-sample.

These are *web-photo* benchmarks of mostly public figures. They are easier than a doorway
camera in some ways (photographers, good lighting) and harder in others (pose, age gaps
between photos). LFW is demographically skewed (about 80% White, 75% male), so minority-group
estimates have wider intervals. Ethnicity labels are classifier *estimates*, not
self-identification, and the four-way split is a coarse proxy. Pose is estimated from YuNet
landmarks.

## Metrics

* Attempt-level **FAR** (stranger accepted) and **FRR** (genuine user rejected). A failure to
  detect a face counts as a rejection. Also EER, ROC AUC and failure-to-acquire rate.
* **FAR is measured on dense stranger sets:** every cross-person image pair within a half,
  up to 7.6 million. That resolves fixed-FAR thresholds at 1%, 0.1% and 0.01%, and gives
  subgroup FAR over all same-group strangers.
* **Policies:** fixed FAR (primary: 0.1%), a cost-based policy minimising `10·FAR + 1·FRR`,
  and the EER point. Subgroups are reported at FAR 0.1% and at the cost-based policy.
* **Uncertainty:** 95% person-level (pigeonhole) bootstrap intervals, which resample people,
  not pairs. They come out 1.1-1.3× wider than naive pair-level intervals.

## Results (LFW, FAR fixed at 0.1%)

| | dlib (2022) | SFace FP32 | SFace INT8 |
|---|---|---|---|
| FRR | 8.67% | 1.53% | 1.73% |
| Worst same-ethnicity FAR | 6.02% (Asian) | 0.46% (Black) | 0.43% (Black) |
| Female / male FRR | 1.0× | 2.5× | 2.0× |
| FRR on 90° profiles (CFP-FP) | 86.5% | 26.8% | 31.1% |

Full results: [`reports/findings.md`](reports/findings.md) and
[`reports/tables.md`](reports/tables.md).

## Ethical considerations and caveats

* Face embeddings are biometric special-category data under UK GDPR Art. 9. The lab stores
  templates encrypted (AES-GCM), keyed by pseudonymous IDs, and the study uses only public
  benchmarks. No face images are committed; figures show aggregates only.
* **Known disparities.** SFace rejects women about twice as often as men and falsely accepts
  female strangers about twice as often, on both LFW and CelebA. Its false accepts against
  Black strangers are about 4× the White rate on LFW. Any deployment should measure these on
  its own captures.
* Results on LFW do not transfer to a specific door camera without re-calibrating on
  consented data from that camera.
* The pipeline has not been tested for spoof resistance on the LFW study, by design: LFW
  images are photographs and would be (correctly) rejected by the PAD screen.
