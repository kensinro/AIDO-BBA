# Expected data layout

Paths are configured in `config.local.json`. A typical layout is:

```text
AIDO-Data/
├── UCSC_XENA/
│   ├── Breast Cancer (BRCA)/
│   │   ├── GE.tsv
│   │   ├── BRCA_stage_groups_from_survival.tsv
│   │   ├── Phenotype.tsv
│   │   └── TCGA.BRCA.sampleMap_BRCA_clinicalMatrix
│   └── Kidney Clear Cell Carcinoma (KIRC)/
│       ├── GE.tsv
│       ├── Phenotype.tsv
│       └── TCGA.KIRC.sampleMap_KIRC_clinicalMatrix
├── External/
│   ├── brca_metabric/
│   │   ├── data_mrna_illumina_microarray.txt
│   │   └── brca_metabric_clinical_data.tsv
│   └── GSE96058/
│       ├── GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv
│       ├── GSE96058-GPL11154_series_matrix.txt
│       └── GSE96058-GPL18573_series_matrix.txt
├── GSEA/c5.go.bp.v2026.1.Hs.symbols.gmt
├── HGNC/hgnc_complete_set.txt
└── NCBI_Gene/Homo_sapiens.gene_info
```

Some source tables may use UTF-16; the KIRC script includes encoding detection. Do not commit controlled, licensed, or large cohort data to GitHub.
