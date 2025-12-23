**📱 Google Play Store App Rating Prediction**
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit‑Learn

**Objective:**
Predict app ratings using metadata from the Google Play Store to help identify high‑potential apps that deserve visibility boosts.

**Process:**
->Loaded and cleaned the Google Play Store dataset (10,000+ app records).

->Handled missing values, corrected data types, and standardized fields (Size, Installs, Price, Reviews).

->Performed sanity checks to remove invalid ratings, inconsistent price values, and impossible review counts.

->Conducted univariate and bivariate analysis to understand rating patterns across price, size, installs, category, and content rating.

->Treated outliers in Price, Reviews, and Installs using percentile thresholds and domain logic.

->Applied log transformation to reduce skew in Reviews and Installs.

->Encoded categorical variables (Category, Genres, Content Rating) using dummy variables.

->Built a Linear Regression model using a 70‑30 train‑test split.

->Evaluated model performance using R² on both training and test sets.

**Impact:**
Enabled identification of app characteristics that strongly influence user ratings, helping stakeholders understand which app types are most promising for promotion and visibility boosts on the Play Store.
