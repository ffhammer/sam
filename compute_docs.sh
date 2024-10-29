

# # create images
# rm -r docs/imgs/*

# python scripts/img_creation/increase_in_lcs_plotly.py --dir4imgs  docs/imgs/increase_in_lcs/
# python scripts/img_creation/dose_response_curves.py --dir4imgs docs/imgs/dose_response_curves/
# python scripts/predict_all.py --plot True --dir4imgs docs/imgs/sam_predictions/


for file in docs/imgs/sam_predictions/*; do 
    mv "$file" "${file// /_}"
done

#  create markdown
rm docs/experiments/*
python scripts/generate_docs/generate_experiments.py

python scripts/generate_docs/render_templates.py

