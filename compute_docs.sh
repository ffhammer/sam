

# # create images
# rm -r docs/imgs/*

# python scripts/img_creation/increase_in_lcs_plotly.py --dir4imgs  docs/imgs/increase_in_lcs/
# python scripts/img_creation/dose_response_curves.py --dir4imgs docs/imgs/dose_response_curves/
# python scripts/predict_all.py --plot True --dir4imgs docs/imgs/sam_predictions/


for file in docs/imgs/sam_predictions/*; do 
    if [[ "$file" == *" "* ]]; then
        mv "$file" "${file// /_}"
    fi
done

# Remove old markdown files
rm docs/experiments/*

# Generate new markdown files
python scripts/generate_docs/generate_experiments.py

# Render templates
python scripts/generate_docs/render_templates.py


#clean
cd docs
rm -r _static _sources
rm *.html objects.inv searchindex.js

cd sphinx
make html
cd ..

mv  _build/html/* ./
rm -r _build