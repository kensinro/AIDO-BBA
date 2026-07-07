# %% CELL 30.1 — Imports and purpose
from pathlib import Path
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('AIDO-BBA RECOMMENDATION-SPECIFICITY AUDIT')
print('=' * 80)

# %% CELL 30.2 — Settings and run discovery
OUTPUT_ROOT = Path(r'D:\AIDO-Temp\AIDO_BBA_BRCA_1_0')
TOP_N_REVISED_TARGETS_PER_PATIENT = 20
TOP_N_DISPLAY_GENES = 40
TOP_N_DISPLAY_PATIENTS = 40
HIGH_GLOBAL_FREQUENCY_FRACTION = 0.25
MODERATE_GLOBAL_FREQUENCY_FRACTION = 0.10
ALPHA_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]
MIN_REPEAT_SELECTION_FREQUENCY = 0.40
MIN_FOLD_SELECTION_FREQUENCY = 0.20

candidate_runs = []
for run_dir in OUTPUT_ROOT.iterdir():
    if not run_dir.is_dir():
        continue
    required_files = [
        run_dir / '16_missing_measurement_recommendation' / 'summaries' / 'patient_gap_gene_measurement_targets.tsv',
        run_dir / '16_missing_measurement_recommendation' / 'summaries' / 'all_patient_measurement_recommendation_report.tsv',
        run_dir / '16_missing_measurement_recommendation' / 'summaries' / 'boundary_patient_measurement_recommendation_top100.tsv',
    ]
    if all(path.exists() for path in required_files):
        candidate_runs.append(run_dir)

if not candidate_runs:
    raise FileNotFoundError('No completed CELL 29/29B recommendation run was found.')

RUN_DIR = sorted(candidate_runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
RECOMMENDATION_DIR = RUN_DIR / '16_missing_measurement_recommendation'
RECOMMENDATION_SUMMARY_DIR = RECOMMENDATION_DIR / 'summaries'
SPECIFICITY_DIR = RUN_DIR / '17_recommendation_specificity_audit'
SPECIFICITY_SUMMARY_DIR = SPECIFICITY_DIR / 'summaries'
SPECIFICITY_REPORT_DIR = SPECIFICITY_DIR / 'patient_reports'
for directory in [SPECIFICITY_DIR, SPECIFICITY_SUMMARY_DIR, SPECIFICITY_REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print('\nRun:')
print(RUN_DIR)
print('\nSpecificity output directory:')
print(SPECIFICITY_DIR)

# %% CELL 30.3 — Load and standardize inputs
targets = pd.read_csv(RECOMMENDATION_SUMMARY_DIR / 'patient_gap_gene_measurement_targets.tsv', sep='\t')
all_patient_reports = pd.read_csv(RECOMMENDATION_SUMMARY_DIR / 'all_patient_measurement_recommendation_report.tsv', sep='\t')
boundary_reports = pd.read_csv(RECOMMENDATION_SUMMARY_DIR / 'boundary_patient_measurement_recommendation_top100.tsv', sep='\t')

def first_existing_column(dataframe, candidates, required=True):
    for column in candidates:
        if column in dataframe.columns:
            return column
    if required:
        raise KeyError('None of the expected columns were found:\n' + '\n'.join(candidates))
    return None

patient_id_column = first_existing_column(targets, ['patient_id'])
raw_gene_column = first_existing_column(targets, ['resolved_raw_gene_id', 'raw_gene_id', 'gene_id'])
harmonized_gene_column = first_existing_column(
    targets,
    ['resolved_harmonized_gene_id', 'harmonized_gene_id_y', 'harmonized_gene_id', 'harmonized_gene_id_x'],
    required=False,
)
core_column = first_existing_column(targets, ['resolved_core_module_name', 'core_module_name'])
base_priority_column = first_existing_column(targets, ['combined_patient_gene_priority', 'gene_measurement_priority_score'])
patient_abs_column = first_existing_column(
    targets,
    ['pipeline_mean_absolute_shap_x', 'pipeline_mean_absolute_shap', 'conditional_mean_absolute_shap'],
    required=False,
)
patient_signed_column = first_existing_column(
    targets,
    ['pipeline_mean_signed_shap_x', 'pipeline_mean_signed_shap', 'conditional_mean_signed_shap'],
    required=False,
)
global_abs_column = first_existing_column(
    targets,
    ['pipeline_mean_absolute_shap_y', 'global_pipeline_mean_absolute_shap', 'pipeline_mean_absolute_shap'],
    required=False,
)

targets['patient_id'] = targets[patient_id_column].astype(str)
targets['raw_gene_id_resolved'] = targets[raw_gene_column].astype(str)
if harmonized_gene_column is None:
    targets['harmonized_gene_id_resolved'] = targets['raw_gene_id_resolved']
else:
    targets['harmonized_gene_id_resolved'] = (
        targets[harmonized_gene_column]
        .where(targets[harmonized_gene_column].notna(), targets['raw_gene_id_resolved'])
        .astype(str)
    )
targets['core_module_name_resolved'] = targets[core_column].astype(str)
targets['base_patient_gene_priority'] = pd.to_numeric(targets[base_priority_column], errors='coerce')
targets['patient_absolute_attribution'] = (
    pd.to_numeric(targets[patient_abs_column], errors='coerce') if patient_abs_column else np.nan
)
targets['patient_signed_attribution'] = (
    pd.to_numeric(targets[patient_signed_column], errors='coerce') if patient_signed_column else np.nan
)
targets['global_absolute_attribution'] = (
    pd.to_numeric(targets[global_abs_column], errors='coerce') if global_abs_column else np.nan
)
for dataframe in [all_patient_reports, boundary_reports]:
    dataframe['patient_id'] = dataframe['patient_id'].astype(str)

print('\nTarget rows:', targets.shape)
print('Unique target patients:', targets['patient_id'].nunique())
print('Unique raw genes:', targets['raw_gene_id_resolved'].nunique())

# %% CELL 30.4 — Global target-frequency audit
n_target_patients = targets['patient_id'].nunique()
global_gene_frequency = (
    targets.groupby(['raw_gene_id_resolved', 'harmonized_gene_id_resolved'], as_index=False)
    .agg(
        n_patient_targets=('patient_id', 'nunique'),
        n_target_rows=('patient_id', 'size'),
        n_cores=('core_module_name_resolved', 'nunique'),
        cores=('core_module_name_resolved', lambda v: ' | '.join(sorted(set(v.astype(str))))),
        mean_base_priority=('base_patient_gene_priority', 'mean'),
        median_base_priority=('base_patient_gene_priority', 'median'),
        mean_patient_absolute_attribution=('patient_absolute_attribution', 'mean'),
        mean_global_absolute_attribution=('global_absolute_attribution', 'mean'),
        mean_repeat_selection_frequency=('repeat_selection_frequency', 'mean'),
        mean_fold_selection_frequency=('fold_selection_frequency', 'mean'),
        mean_sign_consistency=('attribution_sign_consistency', 'mean'),
    )
)
global_gene_frequency['patient_frequency_fraction'] = global_gene_frequency['n_patient_targets'] / n_target_patients
global_gene_frequency['idf_raw'] = np.log(
    (1.0 + n_target_patients) / (1.0 + global_gene_frequency['n_patient_targets'])
) + 1.0
idf_min = global_gene_frequency['idf_raw'].min()
idf_max = global_gene_frequency['idf_raw'].max()
global_gene_frequency['idf_normalized'] = (
    global_gene_frequency['idf_raw'] - idf_min
) / max(idf_max - idf_min, 1e-12)
global_gene_frequency['global_target_class'] = np.select(
    [
        global_gene_frequency['patient_frequency_fraction'] >= HIGH_GLOBAL_FREQUENCY_FRACTION,
        global_gene_frequency['patient_frequency_fraction'] >= MODERATE_GLOBAL_FREQUENCY_FRACTION,
    ],
    ['generic_high_frequency_gap_gene', 'recurrent_gap_gene'],
    default='patient_specific_candidate_gene',
)
global_gene_frequency = global_gene_frequency.sort_values(
    ['n_patient_targets', 'mean_base_priority'], ascending=[False, False]
).reset_index(drop=True)

display(global_gene_frequency.head(TOP_N_DISPLAY_GENES))

# %% CELL 30.5 — Merge specificity into patient-gene targets
targets_specificity = targets.merge(
    global_gene_frequency[
        [
            'raw_gene_id_resolved',
            'harmonized_gene_id_resolved',
            'n_patient_targets',
            'patient_frequency_fraction',
            'idf_raw',
            'idf_normalized',
            'global_target_class',
        ]
    ],
    on=['raw_gene_id_resolved', 'harmonized_gene_id_resolved'],
    how='left',
    validate='many_to_one',
)
targets_specificity['passes_minimum_support'] = (
    (targets_specificity['repeat_selection_frequency'] >= MIN_REPEAT_SELECTION_FREQUENCY)
    & (targets_specificity['fold_selection_frequency'] >= MIN_FOLD_SELECTION_FREQUENCY)
)
targets_specificity['patient_specificity_score'] = targets_specificity['idf_normalized']
targets_specificity['genericity_score'] = targets_specificity['patient_frequency_fraction']
targets_specificity['specificity_adjusted_priority'] = (
    targets_specificity['base_patient_gene_priority']
    * (0.50 + 0.50 * targets_specificity['patient_specificity_score'])
)
targets_specificity['specificity_adjusted_rank_within_patient'] = (
    targets_specificity.groupby('patient_id')['specificity_adjusted_priority']
    .rank(method='first', ascending=False)
    .astype(int)
)
targets_specificity['specificity_adjusted_rank_within_patient_core'] = (
    targets_specificity.groupby(['patient_id', 'core_module_name_resolved'])['specificity_adjusted_priority']
    .rank(method='first', ascending=False)
    .astype(int)
)
targets_specificity['recommendation_specificity_class'] = np.select(
    [
        targets_specificity['global_target_class'].eq('generic_high_frequency_gap_gene'),
        targets_specificity['global_target_class'].eq('recurrent_gap_gene')
        & targets_specificity['specificity_adjusted_rank_within_patient'].le(TOP_N_REVISED_TARGETS_PER_PATIENT),
        targets_specificity['global_target_class'].eq('patient_specific_candidate_gene')
        & targets_specificity['specificity_adjusted_rank_within_patient'].le(TOP_N_REVISED_TARGETS_PER_PATIENT),
    ],
    [
        'generic_background_gap_target',
        'recurrent_patient-relevant_target',
        'patient_specific_priority_target',
    ],
    default='lower_priority_target',
)
revised_targets = targets_specificity[
    targets_specificity['passes_minimum_support']
    & targets_specificity['specificity_adjusted_rank_within_patient'].le(TOP_N_REVISED_TARGETS_PER_PATIENT)
].copy()

print('\nRevised target rows:', revised_targets.shape)
display(
    revised_targets['recommendation_specificity_class']
    .value_counts()
    .rename_axis('recommendation_specificity_class')
    .reset_index(name='n_rows')
)

# %% CELL 30.6 — Patient-level specificity audit
patient_specificity_summary = (
    revised_targets.groupby('patient_id', as_index=False)
    .agg(
        n_revised_targets=('raw_gene_id_resolved', 'size'),
        n_unique_revised_targets=('raw_gene_id_resolved', 'nunique'),
        n_patient_specific_targets=(
            'recommendation_specificity_class',
            lambda v: int(np.sum(v == 'patient_specific_priority_target')),
        ),
        n_recurrent_targets=(
            'recommendation_specificity_class',
            lambda v: int(np.sum(v == 'recurrent_patient-relevant_target')),
        ),
        n_generic_targets=(
            'recommendation_specificity_class',
            lambda v: int(np.sum(v == 'generic_background_gap_target')),
        ),
        mean_patient_specificity_score=('patient_specificity_score', 'mean'),
        median_patient_specificity_score=('patient_specificity_score', 'median'),
        mean_specificity_adjusted_priority=('specificity_adjusted_priority', 'mean'),
        maximum_specificity_adjusted_priority=('specificity_adjusted_priority', 'max'),
        dominant_core=('core_module_name_resolved', lambda v: v.value_counts().index[0]),
    )
)
patient_specificity_summary['patient_specific_target_fraction'] = (
    patient_specificity_summary['n_patient_specific_targets']
    / patient_specificity_summary['n_revised_targets'].replace(0, np.nan)
)
patient_specificity_summary['generic_target_fraction'] = (
    patient_specificity_summary['n_generic_targets']
    / patient_specificity_summary['n_revised_targets'].replace(0, np.nan)
)
patient_specificity_summary['overall_recommendation_specificity_score'] = (
    0.50 * patient_specificity_summary['patient_specific_target_fraction']
    + 0.30 * patient_specificity_summary['mean_patient_specificity_score']
    + 0.20 * (1.0 - patient_specificity_summary['generic_target_fraction'])
)
patient_specificity_summary['recommendation_specificity_tier'] = pd.cut(
    patient_specificity_summary['overall_recommendation_specificity_score'],
    bins=[-np.inf, 0.35, 0.60, np.inf],
    labels=['low_specificity', 'moderate_specificity', 'higher_specificity'],
)
patient_specificity_summary = all_patient_reports.merge(
    patient_specificity_summary,
    on='patient_id',
    how='left',
    validate='one_to_one',
)
for column in [
    'n_revised_targets',
    'n_unique_revised_targets',
    'n_patient_specific_targets',
    'n_recurrent_targets',
    'n_generic_targets',
]:
    patient_specificity_summary[column] = patient_specificity_summary[column].fillna(0).astype(int)

# %% CELL 30.7 — Core-level specificity audit
core_specificity_summary = (
    revised_targets.groupby('core_module_name_resolved', as_index=False)
    .agg(
        n_target_rows=('patient_id', 'size'),
        n_patients=('patient_id', 'nunique'),
        n_unique_genes=('raw_gene_id_resolved', 'nunique'),
        mean_patient_frequency_fraction=('patient_frequency_fraction', 'mean'),
        mean_idf_normalized=('idf_normalized', 'mean'),
        fraction_patient_specific_targets=(
            'recommendation_specificity_class',
            lambda v: float(np.mean(v == 'patient_specific_priority_target')),
        ),
        fraction_generic_targets=(
            'recommendation_specificity_class',
            lambda v: float(np.mean(v == 'generic_background_gap_target')),
        ),
        mean_specificity_adjusted_priority=('specificity_adjusted_priority', 'mean'),
    )
)
core_specificity_summary['core_specificity_score'] = (
    0.60 * core_specificity_summary['mean_idf_normalized']
    + 0.40 * core_specificity_summary['fraction_patient_specific_targets']
)
display(core_specificity_summary.sort_values('core_specificity_score', ascending=False))

# %% CELL 30.8 — Alpha sensitivity analysis
sensitivity_records = []
for alpha in ALPHA_VALUES:
    working = targets_specificity.copy()
    working['alpha_adjusted_priority'] = (
        working['base_patient_gene_priority']
        * ((1.0 - alpha) + alpha * (0.50 + 0.50 * working['idf_normalized']))
    )
    working['alpha_rank_within_patient'] = (
        working.groupby('patient_id')['alpha_adjusted_priority']
        .rank(method='first', ascending=False)
        .astype(int)
    )
    selected = working[
        working['passes_minimum_support']
        & working['alpha_rank_within_patient'].le(TOP_N_REVISED_TARGETS_PER_PATIENT)
    ].copy()
    for patient_id, patient_df in selected.groupby('patient_id'):
        sensitivity_records.append(
            {
                'alpha': alpha,
                'patient_id': patient_id,
                'n_targets': len(patient_df),
                'mean_idf_normalized': float(patient_df['idf_normalized'].mean()),
                'fraction_generic_targets': float(
                    np.mean(patient_df['global_target_class'] == 'generic_high_frequency_gap_gene')
                ),
                'fraction_patient_specific_targets': float(
                    np.mean(patient_df['global_target_class'] == 'patient_specific_candidate_gene')
                ),
            }
        )
sensitivity_patient = pd.DataFrame(sensitivity_records)
sensitivity_summary = (
    sensitivity_patient.groupby('alpha', as_index=False)
    .agg(
        n_patients=('patient_id', 'nunique'),
        mean_targets_per_patient=('n_targets', 'mean'),
        mean_idf_normalized=('mean_idf_normalized', 'mean'),
        mean_fraction_generic_targets=('fraction_generic_targets', 'mean'),
        mean_fraction_patient_specific_targets=('fraction_patient_specific_targets', 'mean'),
    )
)
display(sensitivity_summary)

# %% CELL 30.9 — Build revised target text and patient reports
def build_gene_label(row):
    raw_gene = str(row['raw_gene_id_resolved'])
    harmonized_gene = str(row['harmonized_gene_id_resolved'])
    core_name = str(row['core_module_name_resolved'])
    specificity_class = str(row['recommendation_specificity_class'])
    invalid_values = {'', 'nan', 'none', 'null', 'na'}
    gene_text = (
        f'{raw_gene}->{harmonized_gene}'
        if harmonized_gene.lower() not in invalid_values and harmonized_gene != raw_gene
        else raw_gene
    )
    return f'{gene_text}[{core_name};{specificity_class}]'

revised_targets['specificity_adjusted_gene_label'] = revised_targets.apply(build_gene_label, axis=1)
patient_revised_gene_text_records = []
for patient_id, patient_df in (
    revised_targets.sort_values(['patient_id', 'specificity_adjusted_rank_within_patient'])
    .groupby('patient_id')
):
    patient_revised_gene_text_records.append(
        {
            'patient_id': patient_id,
            'specificity_adjusted_gap_genes': '; '.join(
                patient_df['specificity_adjusted_gene_label'].tolist()
            ),
            'n_specificity_adjusted_gap_genes': len(patient_df),
        }
    )
patient_revised_gene_text = pd.DataFrame(patient_revised_gene_text_records)
final_patient_reports = patient_specificity_summary.merge(
    patient_revised_gene_text,
    on='patient_id',
    how='left',
    validate='one_to_one',
)
final_patient_reports['specificity_adjusted_gap_genes'] = final_patient_reports[
    'specificity_adjusted_gap_genes'
].fillna('')
final_patient_reports['n_specificity_adjusted_gap_genes'] = final_patient_reports[
    'n_specificity_adjusted_gap_genes'
].fillna(0).astype(int)

boundary_rank_map = dict(
    zip(
        boundary_reports['patient_id'].astype(str),
        boundary_reports['boundary_priority_rank'],
    )
)
boundary_specificity_reports = final_patient_reports[
    final_patient_reports['patient_id'].isin(boundary_rank_map)
].copy()
boundary_specificity_reports['boundary_priority_rank'] = boundary_specificity_reports['patient_id'].map(
    boundary_rank_map
)
boundary_specificity_reports = boundary_specificity_reports.sort_values(
    'boundary_priority_rank'
).reset_index(drop=True)

# %% CELL 30.10 — Write specificity-adjusted text reports
n_specificity_reports_written = 0
for _, row in boundary_specificity_reports.iterrows():
    patient_id = str(row['patient_id'])
    score = row.get('overall_recommendation_specificity_score', np.nan)
    score_text = f'{score:.4f}' if pd.notna(score) else 'NA'
    revised_gene_text = str(row.get('specificity_adjusted_gap_genes', '')).strip()
    if not revised_gene_text:
        revised_gene_text = 'No specificity-adjusted target passed the current thresholds.'
    report_lines = [
        'AIDO-BBA SPECIFICITY-ADJUSTED MEASUREMENT AUDIT',
        '=' * 64,
        '',
        f'Patient: {patient_id}',
        f"Clinical group: {row.get('true_group', 'NA')}",
        f"Clinical-molecular rank state: {row.get('clinical_molecular_rank_state', 'NA')}",
        f"Integrated BBA state: {row.get('integrated_bba_state', 'NA')}",
        f"Model-dependence tier: {row.get('model_dependence_tier', 'NA')}",
        f"Repeat-instability tier: {row.get('repeat_instability_tier', 'NA')}",
        '',
        'RECOMMENDATION SPECIFICITY',
        '-' * 64,
        f'Specificity score: {score_text}',
        f"Specificity tier: {row.get('recommendation_specificity_tier', 'NA')}",
        f"Patient-specific targets: {row.get('n_patient_specific_targets', 0)}",
        f"Recurrent targets: {row.get('n_recurrent_targets', 0)}",
        f"Generic targets: {row.get('n_generic_targets', 0)}",
        '',
        'SPECIFICITY-ADJUSTED REPRESENTATION-GAP TARGETS',
        '-' * 64,
        revised_gene_text,
        '',
        'ORIGINAL AUDIT RECOMMENDATION',
        '-' * 64,
        str(row.get('measurement_recommendation_summary', 'NA')),
        '',
        'INTERPRETATION BOUNDARY',
        '-' * 64,
        (
            'This report separates globally recurrent representation gaps from '
            'patient-specific computational measurement priorities. It does not '
            'prescribe clinical testing, establish biomarkers, infer prognosis, '
            'or recommend treatment.'
        ),
    ]
    report_path = SPECIFICITY_REPORT_DIR / f'{patient_id}_specificity_adjusted_measurement_audit.txt'
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    n_specificity_reports_written += 1

# %% CELL 30.11 — Save outputs and manifest
global_gene_frequency.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'global_target_gene_frequency_audit.tsv', sep='\t', index=False
)
targets_specificity.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'all_patient_gene_targets_with_specificity.tsv', sep='\t', index=False
)
revised_targets.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'specificity_adjusted_patient_gene_targets.tsv', sep='\t', index=False
)
patient_specificity_summary.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'patient_recommendation_specificity_summary.tsv', sep='\t', index=False
)
core_specificity_summary.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'core_recommendation_specificity_summary.tsv', sep='\t', index=False
)
sensitivity_patient.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'specificity_weighting_sensitivity_patient.tsv', sep='\t', index=False
)
sensitivity_summary.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'specificity_weighting_sensitivity_summary.tsv', sep='\t', index=False
)
final_patient_reports.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'all_patient_specificity_adjusted_recommendation_report.tsv', sep='\t', index=False
)
boundary_specificity_reports.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'boundary_patient_specificity_adjusted_report_top100.tsv', sep='\t', index=False
)

summary_table = pd.DataFrame(
    [
        {'metric': 'n_target_patients', 'value': n_target_patients},
        {'metric': 'n_original_target_rows', 'value': len(targets)},
        {'metric': 'n_unique_target_genes', 'value': global_gene_frequency['raw_gene_id_resolved'].nunique()},
        {
            'metric': 'n_generic_high_frequency_genes',
            'value': int(
                (global_gene_frequency['global_target_class'] == 'generic_high_frequency_gap_gene').sum()
            ),
        },
        {
            'metric': 'n_recurrent_genes',
            'value': int((global_gene_frequency['global_target_class'] == 'recurrent_gap_gene').sum()),
        },
        {
            'metric': 'n_patient_specific_candidate_genes',
            'value': int(
                (global_gene_frequency['global_target_class'] == 'patient_specific_candidate_gene').sum()
            ),
        },
        {'metric': 'n_specificity_adjusted_target_rows', 'value': len(revised_targets)},
        {
            'metric': 'n_patient_specific_priority_target_rows',
            'value': int(
                (revised_targets['recommendation_specificity_class'] == 'patient_specific_priority_target').sum()
            ),
        },
        {
            'metric': 'n_recurrent_patient_relevant_target_rows',
            'value': int(
                (revised_targets['recommendation_specificity_class'] == 'recurrent_patient-relevant_target').sum()
            ),
        },
        {
            'metric': 'n_generic_background_target_rows',
            'value': int(
                (revised_targets['recommendation_specificity_class'] == 'generic_background_gap_target').sum()
            ),
        },
        {
            'metric': 'specificity_adjusted_boundary_reports_written',
            'value': n_specificity_reports_written,
        },
    ]
)
summary_table.to_csv(
    SPECIFICITY_SUMMARY_DIR / 'recommendation_specificity_audit_summary.tsv', sep='\t', index=False
)
manifest = {
    'analysis': 'AIDO-BBA recommendation-specificity audit',
    'run_directory': str(RUN_DIR),
    'n_target_patients': int(n_target_patients),
    'high_global_frequency_fraction': HIGH_GLOBAL_FREQUENCY_FRACTION,
    'moderate_global_frequency_fraction': MODERATE_GLOBAL_FREQUENCY_FRACTION,
    'top_revised_targets_per_patient': TOP_N_REVISED_TARGETS_PER_PATIENT,
    'specificity_adjusted_priority': 'base priority × (0.5 + 0.5 × normalized IDF)',
    'interpretation_boundary': (
        'Specificity-adjusted targets are computational measurement priorities, '
        'not validated biomarkers or clinical prescriptions.'
    ),
}
with open(SPECIFICITY_DIR / 'recommendation_specificity_audit_manifest.json', 'w', encoding='utf-8') as handle:
    json.dump(manifest, handle, indent=2)

# %% CELL 30.12 — Final display
print('\n' + '=' * 80)
print('CELL 30 COMPLETED')
print('=' * 80)
display(summary_table)
print('\nMost globally recurrent recommendation genes:')
display(global_gene_frequency.head(TOP_N_DISPLAY_GENES))
print('\nCore specificity summary:')
display(core_specificity_summary.sort_values('core_specificity_score', ascending=False))
print('\nSpecificity weighting sensitivity:')
display(sensitivity_summary)
print('\nTop boundary patients after specificity adjustment:')
display(boundary_specificity_reports.head(TOP_N_DISPLAY_PATIENTS))
print('\nOutput directory:')
print(SPECIFICITY_DIR)
print('\nSpecificity-adjusted individual reports:')
print(n_specificity_reports_written)
