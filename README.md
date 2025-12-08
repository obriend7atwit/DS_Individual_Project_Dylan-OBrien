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
  - Placeholder (example):  
    ```md
    <img width="3568" height="1768" alt="figure_1_Distribution_of_Taste_Labels_Primary_Secondary" src="https://github.com/user-attachments/assets/1278671e-7e50-4fe6-bd6b-ef6eac83fb8e" />
    ```

- **Taste vs difficulty heatmap**  
  - Plot: heatmap of taste × difficulty, row-normalized (easy → medium → hard).  
  - Notebook: **Cell 11** — taste–difficulty relationship.  
  - Placeholder:  
    ```md
    ![Taste vs Difficulty Heatmap](graphs/figure_2_Taste_vs_Difficulty.png)  <!-- Output of Cell 11 -->
    ```

- **Per-taste F1 bar chart**  
  - Plot: bar chart of per-label F1-scores for the multi-label taste model.  
  - Notebook: **Cell 17** — per-taste F1 scores.  
  - Placeholder:  
    ```md
    ![Per-Taste F1 Scores](graphs/figure_3_Per_Taste_F1_Scores.png)  <!-- Output of Cell 17 -->
    ```

- **Top ingredients per taste (table)**  
  - Table: top indicative tokens/ingredients per taste (e.g., top 15 per taste).  
  - Notebook: **Cell 18** — top indicative tokens per taste label.  
  - Placeholder:  
    > 📋 **Table:** Top 15 tokens per taste (screenshot or exported table from Cell 18).

---

### Q2. What recipe features are most linked to shorter or longer prep times?

**Short answer:**  
Prep time increases with both **number of ingredients** and **number of steps**. The dataset shows:

- Recipes with many ingredients and steps tend to have longer prep times.  
- Cook time also correlates with prep time but somewhat less strongly.  
- Simple, low-ingredient recipes cluster at short prep times, while long, multi-step recipes form a heavy tail of “project meals”.

Correlation analysis highlights `num_steps` and `num_ingredients` as the strongest structural predictors of `est_prep_time_min`.

**Suggested artifacts to embed**

- **Distributions of structural features**  
  - Plot: histograms/boxplots of `num_ingredients` and `num_steps`.  
  - Notebook: **Cell 9** — numeric distributions for ingredients and steps.  
  - Placeholder:  
    ```md
    ![Number of Ingredients and Steps](graphs/figure_4_Num_Ingredients_and_Steps.png)  <!-- Output of Cell 9 -->
    ```

- **Prep and cook time distributions (raw minutes)**  
  - Plot: histograms/boxplots of `est_prep_time_min` and `est_cook_time_min` (non-log).  
  - Notebook: **Cell 10** — raw prep and cook time distributions.  
  - Placeholder:  
    ```md
    ![Prep and Cook Time Distributions](graphs/figure_5_Prep_and_Cook_Time_Distributions.png)  <!-- Output of Cell 10 -->
    ```

- **Prep & cook time by difficulty**  
  - Plot: boxplots of prep and cook time split by difficulty.  
  - Notebook: **Cell 12** — prep/cook time by difficulty.  
  - Placeholder:  
    ```md
    ![Prep and Cook Time by Difficulty](graphs/figure_6_Prep_and_Cook_Time_by_Difficulty.png)  <!-- Output of Cell 12 -->
    ```

- **Structural complexity vs total time**  
  - Plot: scatter plot of total time vs structural features (`num_steps`, `num_ingredients`), colored by difficulty.  
  - Notebook: **Cell 14** — structural complexity vs total time.  
  - Placeholder:  
    ```md
    ![Structural Complexity vs Total Time](graphs/figure_7_Structural_Complexity_vs_Total_Time.png)  <!-- Output of Cell 14 -->
    ```

- **Correlation bar chart**  
  - Plot: bar chart of absolute Pearson correlations between `num_ingredients`, `num_steps`, `est_cook_time_min` and `est_prep_time_min`.  
  - Notebook: **Cell 41** — Q2 correlations with prep time.  
  - Placeholder:  
    ```md
    ![Correlation with Prep Time](graphs/figure_8_Correlation_with_Prep_Time.png)  <!-- Output of Cell 41 -->
    ```

---

### Q3. What factors explain difficulty level, and how can beginner-friendly recipes be spotted?

**Short answer:**  
Difficulty is driven by **structural complexity** and **instruction text**:

- More ingredients, more steps, and longer prep/total time push recipes toward “medium” and “hard”.  
- Technique verbs (“whisk”, “sear”, “deglaze”, “proof”) and specialized equipment terms tend to indicate higher difficulty.  
- A text-only classifier performs reasonably well; augmenting it with structural features improves accuracy and reduces confusion.

Beginner-friendly recipes are those with short ingredient lists, few steps, short prep time, and high predicted probability of “easy”.

**Suggested artifacts to embed**

- **Difficulty distribution**  
  - Plot: bar chart of difficulty levels (easy / medium / hard).  
  - Notebook: **Cell 8** — difficulty distribution.  
  - Placeholder:  
    ```md
    ![Distribution of Difficulty Levels](graphs/figure_9_Distribution_of_Difficulty_Levels.png)  <!-- Output of Cell 8 -->
    ```

- **Prep & cook time by difficulty**  
  - Already referenced in Q2 (Cell 12); may be re-used here for context.

- **Text-only difficulty confusion matrix**  
  - Plot: normalized confusion matrix heatmap for the text-only difficulty model.  
  - Notebook: **Cell 21** — confusion matrix (text-only model).  
  - Placeholder:  
    ```md
    ![Difficulty Confusion Matrix (Text-Only)](graphs/figure_10_Difficulty_CM_Text_Only.png)  <!-- Output of Cell 21 -->
    ```

- **Text vs text+structural comparison**  
  - Plot: side-by-side confusion matrices for text-only (red) vs text+structural (green) difficulty models.  
  - Notebook: **Cell 26** — comparison of difficulty models.  
  - Placeholder:  
    ```md
    ![Difficulty Models: Text vs Text+Structural](graphs/figure_11_Difficulty_Models_Comparison.png)  <!-- Output of Cell 26 -->
    ```

- **Numeric feature importance for structural features**  
  - Plot: bar chart of coefficients/importances for numeric features (e.g., `num_steps`, `num_ingredients`, times).  
  - Notebook: **Cell 25** — numeric feature coefficients for difficulty.  
  - Placeholder:  
    ```md
    ![Structural Feature Importance for Difficulty](graphs/figure_12_Difficulty_Structural_Features.png)  <!-- Output of Cell 25 -->
    ```

---

### Q4. Can cuisine type be predicted from ingredients and short instructions?

**Short answer:**  
Cuisine is partly predictable from ingredients and instructions:

- A TF–IDF + Logistic Regression classifier on combined ingredient/instruction text can distinguish common cuisines (e.g., American, Italian, Indian, Mexican, British) with reasonable accuracy.  
- Confusions often follow culinary similarity (e.g., Mediterranean vs Middle Eastern vs British when many dishes share overlapping ingredients).  
- Restricting to the **top 5 cuisines** by frequency produces a cleaner confusion matrix and highlights which cuisines are most distinct in the dataset.

**Suggested artifacts to embed**

- **Top cuisines by recipe count**  
  - Plot: bar chart of the most frequent cuisines.  
  - Notebook: **Cell 13** — top cuisines by recipe count.  
  - Placeholder:  
    ```md
    ![Top Cuisines by Recipe Count](graphs/figure_13_Top_Cuisines.png)  <!-- Output of Cell 13 -->
    ```

- **Cuisine prediction confusion matrix (top-5)**  
  - Plot: confusion matrix focused on the top-5 cuisines.  
  - Notebook: **Cell 44** — Q4 cuisine prediction confusion matrix.  
  - Placeholder:  
    ```md
    ![Cuisine Prediction Confusion Matrix (Top-5)](graphs/figure_14_Cuisine_CM_Top5.png)  <!-- Output of Cell 44 -->
    ```

You may also include an earlier “all-cuisines” confusion matrix for comparison if you saved it.

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

**Suggested artifacts to embed (non-GUI)**

- **Coverage & overlap bar chart (example pantry)**  
  - Plot: bar chart where each bar or group shows coverage (%) and overlap count per recommended recipe.  
  - Notebook: **Cell 30b** — coverage & overlap bar chart.  
  - Placeholder:  
    ```md
    ![Coverage & Overlap for Example Pantry](graphs/figure_15_Coverage_Overlap_Example_Pantry.png)  <!-- Output of Cell 30b -->
    ```

- **Ingredient-space neighborhood (PCA)**  
  - Plot: 2-D PCA showing user pantry (star) and recommended recipes (points) in ingredient space, colored by coverage.  
  - Notebook: **Cell 30c** — ingredient-space neighborhood.  
  - Placeholder:  
    ```md
    ![Ingredient-Space Neighborhood (PCA)](graphs/figure_16_Ingredient_Space_PCA.png)  <!-- Output of Cell 30c -->
    ```

- **Summary charts – predicted difficulty & top tastes**  
  - Plots:  
    - Bar chart of recommended recipes by difficulty.  
    - Bar chart/stacked chart of dominant tastes among recommendations.  
  - Notebook: **Cell 30d** — summary charts for difficulty and tastes.  
  - Placeholder:  
    ```md
    ![Recommended Recipes by Difficulty](graphs/figure_17_Recs_By_Difficulty.png)  <!-- Output of Cell 30d -->
    ![Top Predicted Tastes in Recommendations](graphs/figure_18_Recs_Taste_Profile.png)  <!-- Output of Cell 30d -->
    ```

You can then reference the **interactive GUI** (Cells 33a–33b) with one or two screenshots, for example:

> 🖼️ **Optional GUI screenshots**  
> - Overall GUI layout (pantry input, filters, “Recommend recipes” button).  
> - Recipe details view (description, ingredients, directions) for one recommendation.

These GUI figures don’t need to be auto-saved; simple screenshots from a running notebook are fine.

---

## 7. Final Notes

- Make sure the `graphs/` directory contains the figures you reference in this README, and adjust filenames if your saving scheme differs.  
- Keep the README consistent with the final written report (`report/`), which should follow the course’s Word template and IMRAD structure.  
- For a portfolio or personal site, you can create a shorter summary “project card” linking back to this repo and the full report.
