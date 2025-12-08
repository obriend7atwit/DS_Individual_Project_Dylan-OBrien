# From Pantry to Plate: Predicting Recipe Properties from Ingredients and Instructions

*A DATA 6150 Individual Project using the “Extended Recipes Dataset: 64K Dishes”*

---

## 1. Project Overview

This repository contains a full end-to-end data science project for **DATA 6150 – Data Science Foundations (Fall 2025)** at Wentworth Institute of Technology. The goal is to understand how **ingredient lists** and **short instructions** relate to key recipe properties (taste, prep time, difficulty, cuisine), and to build a **pantry-aware recipe recommender** that helps decide what to cook with ingredients already on hand.

The core analytical questions are:

1. Which ingredients best predict a recipe’s taste profile (sweet, savory, spicy, sour, umami)?  
2. What recipe features (number of ingredients, number of steps, technique words) are most linked to shorter or longer prep times?  
3. What factors explain difficulty level (easy/medium/hard), and how can “beginner-friendly” or “quick-and-simple” recipes be identified?  
4. Can cuisine type be predicted from just the ingredient list and some instructions?  
5. Given a set of pantry items, how can recipes be ranked to use most of what is there while matching target tastes, time, and difficulty?  

The project follows the course rubric: problem definition, data acquisition, EDA, modeling, interpretation, and a final written report using the provided Word template.

---

## 2. Dataset

The analysis uses the **Extended Recipes Dataset: 64K Dishes** (Kaggle), which augments an earlier recipes dataset with rich annotations, including:

- Text fields: title, description, ingredients, directions, cleaned ingredient/instruction text.  
- Structural fields: `num_ingredients`, `num_steps`, `est_prep_time_min`, `est_cook_time_min`.  
- Taste/cuisine: `tastes`, `primary_taste`, `secondary_taste`, `cuisine_list`, `course_list`.  
- Dietary & health: dietary flags (e.g., `is_vegan`, `is_gluten_free`), `dietary_profile`, `healthiness_score`, `health_flags`, `health_level`.

The raw CSV (e.g., `recipes_extended.csv`) is stored under `data/` in the repo (either the full dataset or a filtered/cleaned subset, depending on size constraints).

This is the link to the Kaggle dataset: https://www.kaggle.com/datasets/wafaaelhusseini/extended-recipes-dataset-64k-dishes/data

---

## 3. Methods

High-level methods (explained in more detail in the written report and notebook):

- **EDA & Visualization**
  - Distribution plots for tastes, difficulty, times, and cuisines.
  - Heatmaps relating taste and difficulty.
  - Scatter/box plots linking structural complexity to prep time and total time.

- **NLP + ML**
  - Text cleaning on ingredients and directions.
  - TF–IDF vectorization of ingredient/instruction text.
  - Logistic Regression and tree-based models for:
    - **Taste prediction** (multi-label).
    - **Difficulty prediction** (text-only vs text + structural features).
    - **Cuisine prediction** (multi-class).

- **Pantry-based Recommender**
  - Multi-hot and TF–IDF ingredient vectors.
  - Cosine similarity between pantry and recipes.
  - Multi-objective ranking using **coverage** (how much of a recipe is cookable) and **overlap** (how much of the pantry is used).
  - Optional filters for difficulty, tastes, and time.
  - Static example workflow + interactive GUI (ipywidgets).

---

## 4. Repository Structure

A suggested layout for the GitHub repository:

```text
.
├── README.md                           # This file
├── data/
│   └── recipes_extended.csv            # Or a cleaned / sampled version
├── codes/
│   └── IndividualProject.ipynb         # Main Jupyter notebook
├── graphs/
│   ├── figure_1_Distribution_of_Taste_Labels.png
│   ├── figure_2_Distribution_of_Difficulty_Levels.png
│   ├── ...                             # Other saved figures from the notebook
└── report/
    ├── Dylan_OBrien_Individual_Project_Report.docx
    └── Dylan_OBrien_Individual_Project_Report.pdf
```

---

## 5. Environment & Running the Notebook

**Dependencies (typical):**

- Python 3.10+  
- `pandas`, `numpy`, `scikit-learn`  
- `matplotlib`, `seaborn`  
- `ipywidgets` (for the GUI)  
- `scipy` (for sparse matrices)  

**Example setup (conda):**

```bash
conda create -n recipes-ml python=3.10
conda activate recipes-ml
pip install pandas numpy scikit-learn matplotlib seaborn ipywidgets scipy
jupyter notebook
```

Then:

1. Place `recipes_extended.csv` under `data/`.  
2. Open `codes/IndividualProject.ipynb`.  
3. Run all cells in order.  
   - Early cells: loading, cleaning, and EDA.  
   - Middle cells: taste, difficulty, and cuisine models.  
   - Later cells: pantry recommender and GUI.

Most figures are also saved automatically to `graphs/` (e.g., `figure_#_<Title>.png`) using `plt.savefig`. If you add new plots, follow the same pattern.

---

## 6. Research Questions & Supporting Figures

The following section is designed to act as a narrative summary of the five core questions.

---

### Q1. Which ingredients best predict a recipe’s taste profile?

**Short answer:**  
Taste is strongly encoded in ingredients. Multi-label models using TF–IDF features over ingredients and simple text can reliably predict whether a recipe is **sweet, savory/umami, spicy, sour, bitter, or neutral**. The top ingredients per taste match intuition:

- **Sweet**: sugar, brown sugar, honey, vanilla, chocolate, syrup, cinnamon.  
- **Savory / Umami**: garlic, onion, butter, olive oil, cheese, soy sauce, broth, parmesan.  
- **Spicy**: chili powder, cayenne, jalapeño, red pepper flakes, hot sauce, curry powder.  
- **Sour**: lemon juice, lime juice, vinegar, yogurt, sour cream, buttermilk.

Model coefficients and token tables show that tastes are driven by a relatively small set of strongly identifying ingredients.

- **Taste distribution (combined primary + secondary)**  
  - Plot: bar chart of taste label frequencies.  
  - Notebook: **Cell 7** — “Distribution of taste labels (primary_taste + secondary_taste)”.  
    <img width="3568" height="1768" alt="figure_1_Distribution_of_Taste_Labels_Primary_Secondary" src="https://github.com/user-attachments/assets/1278671e-7e50-4fe6-bd6b-ef6eac83fb8e" />

- **Taste vs difficulty heatmap**  
  - Plot: heatmap of taste × difficulty, row-normalized (easy → medium → hard).  
  - Notebook: **Cell 11** — taste–difficulty relationship.  
    <img width="2536" height="1768" alt="figure_5_Taste_Primary_Secondary_vs_Difficulty_Row_wise" src="https://github.com/user-attachments/assets/c44c7cb0-4586-4cc3-9ab0-50bbb9825041" />



- **Per-taste F1 bar chart**  
  - Plot: bar chart of per-label F1-scores for the multi-label taste model.  
  - Notebook: **Cell 17** — per-taste F1 scores.  
    <img width="3568" height="1768" alt="figure_9_Per_Taste_F1_scores_Multi_Label_Taste_Prediction" src="https://github.com/user-attachments/assets/d503f699-9d05-4311-a606-9063055385d6" />



- **Top ingredients per taste (table)**  
  - Table: top indicative tokens/ingredients per taste (e.g., top 15 per taste).  
  - Notebook: **Cell 18** — top indicative tokens per taste label.  
    <img width="1923" height="288" alt="table1" src="https://github.com/user-attachments/assets/cd20e08e-83dc-4f85-986e-163f52adfef6" />


---

### Q2. What recipe features are most linked to shorter or longer prep times?

**Short answer:**  
Prep time increases with both **number of ingredients** and **number of steps**. The dataset shows:

- Recipes with many ingredients and steps tend to have longer prep times.  
- Cook time also correlates with prep time but somewhat less strongly.  
- Simple, low-ingredient recipes cluster at short prep times, while long, multi-step recipes form a heavy tail of “project meals”.

Correlation analysis highlights `num_steps` and `num_ingredients` as the strongest structural predictors of `est_prep_time_min`.

- **Distributions of structural features**  
  - Plot: histograms/boxplots of `num_ingredients` and `num_steps`.  
  - Notebook: **Cell 9** — numeric distributions for ingredients and steps.  
    <img width="3568" height="1468" alt="figure_3_Distribution_of_Number_of_Steps" src="https://github.com/user-attachments/assets/63418ab7-a22c-41d6-a8f6-ed5eb4d963aa" />


- **Prep and cook time distributions (raw minutes)**  
  - Plot: histograms/boxplots of `est_prep_time_min` and `est_cook_time_min` (non-log).  
  - Notebook: **Cell 10** — raw prep and cook time distributions.  
    <img width="4168" height="1468" alt="figure_4_Cook_Time_Distribution_Minutes" src="https://github.com/user-attachments/assets/7eb64a2d-7566-40b4-8f61-7746f96b2e1c" />


- **Prep & cook time by difficulty**  
  - Plot: boxplots of prep and cook time split by difficulty.  
  - Notebook: **Cell 12** — prep/cook time by difficulty.  
    <img width="4168" height="1768" alt="figure_6_Cook_Time_by_Difficulty_Clipped_at_99th_Percentile" src="https://github.com/user-attachments/assets/80de5282-c525-4655-91e4-a200ccad7c4f" />


- **Structural complexity vs total time**  
  - Plot: scatter plot of total time vs structural features (`num_steps`, `num_ingredients`), colored by difficulty.  
  - Notebook: **Cell 14** — structural complexity vs total time.  
    <img width="3568" height="2068" alt="figure_8_Difficulty" src="https://github.com/user-attachments/assets/bbfccfa9-64ca-48ea-acd4-ead32d615069" />


- **Correlation bar chart**  
  - Plot: bar chart of absolute Pearson correlations between `num_ingredients`, `num_steps`, and `est_cook_time_min`.  
  - Notebook: **Cell Q2** — Q2 correlations with prep time.  
    <img width="2368" height="1468" alt="figure_21_How_structural_features_relate_to_prep_time_correlation_with_est_prep_time_min" src="https://github.com/user-attachments/assets/479623e7-cd3c-4e88-a79d-ea6d082b0c1c" />


---

### Q3. What factors explain difficulty level, and how can beginner-friendly recipes be spotted?

**Short answer:**  
Difficulty is driven by **structural complexity** and **instruction text**:

- More ingredients, more steps, and longer prep/total time push recipes toward “medium” and “hard”.  
- Technique verbs (“whisk”, “sear”, “deglaze”, “proof”) and specialized equipment terms tend to indicate higher difficulty.  
- A text-only classifier performs reasonably well; augmenting it with structural features improves accuracy and reduces confusion.

Beginner-friendly recipes are those with short ingredient lists, few steps, short prep time, and high predicted probability of “easy”.

- **Difficulty distribution**  
  - Plot: bar chart of difficulty levels (easy / medium / hard).  
  - Notebook: **Cell 8** — difficulty distribution.  
    <img width="2368" height="1768" alt="figure_2_Distribution_of_Recipe_Difficulty" src="https://github.com/user-attachments/assets/b0e934c2-d459-4d0c-a42b-8aa996855ab0" />


- **Text vs text+structural comparison**  
  - Plot: side-by-side confusion matrices for text-only (red) vs text+structural (green) difficulty models.  
  - Notebook: **Cell 26** — comparison of difficulty models.  
    <img width="4126" height="1468" alt="figure_12_Text_Structural_Model_Row_wise" src="https://github.com/user-attachments/assets/a81f0a4a-dfb5-40e4-a971-3679dcc5aa45" />


- **Numeric feature importance for structural features**  
  - Plot: bar chart of coefficients/importances for numeric features (e.g., `num_steps`, `num_ingredients`, times).  
  - Notebook: **Cell 25** — numeric feature coefficients for difficulty.  
    <img width="2968" height="1767" alt="figure_11_Difficulty" src="https://github.com/user-attachments/assets/98e9987f-bfbe-497c-bc09-c5befd70cd3e" />


---

### Q4. Can cuisine type be predicted from ingredients and short instructions?

**Short answer:**  
Cuisine is partly predictable from ingredients and instructions:

- A TF–IDF + Logistic Regression classifier on combined ingredient/instruction text can distinguish common cuisines (e.g., American, Italian, Indian, Mexican, British) with reasonable accuracy.  
- Confusions often follow culinary similarity (e.g., Mediterranean vs Middle Eastern vs British when many dishes share overlapping ingredients).  
- Restricting to the **top 5 cuisines** by frequency produces a cleaner confusion matrix and highlights which cuisines are most distinct in the dataset.


- **Top cuisines by recipe count**  
  - Plot: bar chart of the most frequent cuisines.  
  - Notebook: **Cell 13** — top cuisines by recipe count.  
    <img width="3568" height="1769" alt="figure_13_Top_TOP_N_ING_Cleaned_Canonical_Ingredients_by_Recipe_Count_Final" src="https://github.com/user-attachments/assets/714215f3-3502-42b5-b046-eb1f1a2ccdc4" />


- **Cuisine prediction confusion matrix (top-5)**  
  - Plot: confusion matrix focused on the top-5 cuisines.  
  - Notebook: **Cell Q4** — Q4 cuisine prediction confusion matrix.  
    <img width="2268" height="2051" alt="figure_22_Cuisine_Prediction_Confusion_Matrix" src="https://github.com/user-attachments/assets/53e9f446-2e08-4d2f-b098-7602c6593670" />



---

### Q5. Pantry-based recommender: how to rank recipes given pantry items?

**Short answer:**  
The pantry recommender ranks recipes using ingredient similarity and learned recipe attributes:

1. **Ingredient matching**  
   - Each recipe is represented as a multi-hot / TF–IDF ingredient vector.  
   - A user’s pantry is converted into the same space.  
   - For each recipe, the system computes:  
     - **Overlap**: number of matching ingredients between pantry and recipe.  
     - **Coverage**: fraction of recipe ingredients covered by the pantry.

2. **Ranking objectives**  
   - **Coverage mode**: prioritizes recipes that are almost completely cookable.  
   - **Overlap mode**: prioritizes recipes that use as many pantry items as possible.

3. **Taste and difficulty overlays**  
   - The taste and difficulty models are applied to each candidate recipe.  
   - Recommendations display predicted difficulty and dominant tastes, helping choose between quick/easy vs more involved dishes.

A static example pantry is used in the notebook to illustrate the ranking pipeline, and an interactive GUI (ipywidgets) allows arbitrary user input and filters.


- **Coverage & overlap bar chart (example pantry)**  
  - Plot: bar chart where each bar or group shows coverage (%) and overlap count per recommended recipe.  
  - Notebook: **Cell 30b** — coverage & overlap bar chart.  
    <img width="3530" height="2066" alt="figure_14_Pantry_Based_Recommendations_Coverage_Overlap" src="https://github.com/user-attachments/assets/3322a087-383f-4622-b7fd-f9d3a55e6f05" />


- **Ingredient-space neighborhood (PCA)**  
  - Plot: 2-D PCA showing user pantry (star) and recommended recipes (points) in ingredient space, colored by coverage.  
  - Notebook: **Cell 30c** — ingredient-space neighborhood.  
    <img width="2560" height="2068" alt="figure_15_Ingredient_Space_Neighborhood_of_User_Pantry" src="https://github.com/user-attachments/assets/7a45626a-5c00-4c82-be2c-f88046e6ca11" />


- **Summary charts – predicted difficulty & top tastes**  
  - Plots:  
    - Bar chart of recommended recipes by difficulty.  
    - Bar chart/stacked chart of dominant tastes among recommendations.  
  - Notebook: **Cell 30d** — summary charts for difficulty and tastes.  
    <img width="4168" height="1468" alt="figure_16_Top_Predicted_Tastes_top_recommendations" src="https://github.com/user-attachments/assets/b87e6f17-bc40-4352-a4f3-52003909a41e" />



> **GUI screenshots**  
- Overall GUI layout (pantry input, filters, “Recommend recipes” button).
<img width="1547" height="1072" alt="GUI" src="https://github.com/user-attachments/assets/c65b1180-3f71-4d43-ac06-35cf4da6f38a" />

- Recipe details view (description, ingredients, directions) for one recommendation.
<img width="1417" height="542" alt="GUI2" src="https://github.com/user-attachments/assets/f8b9311a-ee0a-4d8f-af2f-a24258d025f8" />


---

