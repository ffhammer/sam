export SAM_REPO_PATH="$(pwd)"


if ! python scripts/img_creation/increase_in_lcs_plotly.py --dir4imgs docs/imgs/increase_in_lcs/; then
    echo "Error: Failed to generate increase in LCs plots with increase_in_lcs_plotly.py."
    exit 1
fi

if ! python scripts/img_creation/dose_response_curves.py --dir4imgs docs/imgs/dose_response_curves/; then
    echo "Error: Failed to create dose-response curves with dose_response_curves.py."
    exit 1
fi

if ! python scripts/predict_all.py --plot True --dir4imgs docs/imgs/sam_predictions/; then
    echo "Error: Failed to run predictions and generate SAM plots with predict_all.py."
    exit 1
fi



for file in docs/imgs/sam_predictions/*; do 
    if [[ "$file" == *" "* ]]; then
        mv "$file" "${file// /_}"
    fi
done


# Generate new markdown files
python scripts/generate_docs/generate_experiments.py

# Render templates
python scripts/generate_docs/render_templates.py

