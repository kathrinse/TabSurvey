#!/bin/bash

N_TRIALS=2

# "LinearModel" "KNN" "DecisionTree" "RandomForest"
# "XGBoost" "CatBoost" "LightGBM"
# "MLP" "TabNet" "VIME"
MODELS=("LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME")
CONFIGS=("config/california_housing.yml" "config/adult.yml" "config/covertype.yml") #

# Some models take forever for the classification dataset...
EXCEPT=("TabNet") # "KNN"

#echo "${MODELS[@]}"
#echo "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do

  for model in "${MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s\n\n' "$model" "$config"

    if [[ "$config" == "config/covertype.yml" ]] && [[ "${EXCEPT[*]}" =~ ${model} ]]; then
      printf "Not executing this..."
      continue
    fi

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs 100

  done

done
