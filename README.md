| Model NAME | Model TYPE | Dataset | Input parameters | Performance (AUC) | Notes |
|---|---|---|---|---|---|
| xgbc1.py | XGBClassifier | n=27088 | 129 reference features | Avg: 0.8497 SD: 0.0052 | regress_lang_time cat. paras need to be converted into a single numerical para, then standardised |
| | XGB based standardisation | ref129_latest.csv (input) | SCQ, RBSR, DCDQ |  | XGB based imputation |
| xgbc2_1.py | | | SCQ only |  |  |
| xgbc2_2.py | | | SCQ selected parameters |  |  |
| xgbc3.py | | n=2637 | CBCL only; 164 cbcl parameters; 394 (after encoding) | Average: 0.9033 SD: 0.0139 | intersection of deriv_cog_impair and cbcl. train model on this data. predict entries for which cbcl is available, but deriv_cog_impair not available (only 3 such entries) |
| xgbc3_1.py | | | CBCL selected parameters |  |  |
| SVC |  | imputation: |  |  |  |
| Random Forest |  | imputation: |  |  |  |
