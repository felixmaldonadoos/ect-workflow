#!/bin/bash


### SEX as FIXED effect ###
bash run_series_mixed_effects_with_aic_submit.sh \
  age_continuous_sex_fixed --sex-effect fixed

### SEX as FIXED effect ###
bash run_series_mixed_effects_with_aic_submit.sh \
  age_continuous_sex_random --sex-effect random