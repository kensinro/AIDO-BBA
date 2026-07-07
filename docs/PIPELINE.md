# Pipeline and audit contract

## Primary sequence

| Stage | Purpose | Main output layer |
|---|---|---|
| Input and endpoint audit | Validate files, TCGA barcodes, primary tumours, and stage labels | sample and endpoint manifests |
| Model ensemble | Repeated 5×5 CV for elastic-net logistic regression and ExtraTrees | fold and patient OOF predictions |
| Attribution audit | Held-out TreeSHAP with additivity and repeat checks | patient–gene attribution tables |
| Explanatory reconstruction | Map attribution to eligible GO-BP sets while retaining an exact residual | patient–process and completeness tables |
| Gap-state audit | Mapping nulls, gap genes, stable cores, fuzzy memberships, and boundary patients | gap/core/state audit tables |
| Measurement triage | Separate cohort sentinels, recurrent gaps, and lower-frequency patient-ranked candidates | patient reports and target lists |
| Transfer stress | Dataset, endpoint, and cancer-type replacement | external stress summaries |

## Interpretation boundary

The pipeline audits what a fitted classifier explains, fails to explain, or cannot support. It does not establish causal mechanisms, molecular subtypes, treatment effects, or clinical utility.
