export SAM_REPO_PATH="$(pwd)"


if ! python docs/generation_scripts/img_creation/increase_in_lcs_plotly.py --dir4imgs docs/imgs/increase_in_lcs/; then
    echo "Error: Failed to generate increase in LCs plots with increase_in_lcs_plotly.py."
    exit 1
fi

if ! python docs/generation_scripts/img_creation/lcs_normed_effect_range.py --dir4imgs docs/imgs/increase_in_lcs/; then
    echo "Error: Failed to generate increase in LCs plots with lcs_normed_effect_range.py."
    exit 1
fi

if ! python docs/generation_scripts/img_creation/dose_response_curves.py --dir4imgs docs/imgs/dose_response_curves/; then
    echo "Error: Failed to create dose-response curves with dose_response_curves.py."
    exit 1
fi

if ! python docs/generation_scripts/predict_all.py --plot True --dir4imgs docs/imgs/sam_predictions/; then
    echo "Error: Failed to run predictions and generate SAM plots with predict_all.py."
    exit 1
fi

if ! python docs/generation_scripts/img_creation/sys_adjusted_imgs.py --dir4imgs docs/imgs/sys_adjusted/; then
    echo "Error: Failed to run predictions and generate SAM plots with sys_adjusted_imgs.py."
    exit 1
fi

if ! python docs/generation_scripts/generate_docs/ecx_sys.py --dir4imgs docs/imgs/exc_sys/; then
    echo "Error: Failed to create ecx sys page."
    exit 1
fi

if ! python docs/generation_scripts/generate_docs/add_sys_page.py --dir4imgs docs/imgs/sys_adjusted/; then
    echo "Error: Failed to build sys page."
    exit 1
fi

for file in docs/imgs/sam_predictions/*; do 
    if [[ "$file" == *" "* ]]; then
        mv "$file" "${file// /_}"
    fi
done


# Generate new markdown files
python docs/generation_scripts/generate_docs/generate_experiments.py

# Render templates
python docs/generation_scripts/generate_docs/render_templates.py

