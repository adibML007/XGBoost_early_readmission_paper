This is the code repository for the paper [DOI to be entered here]

Request access to the dataset here: [text](https://physionet.org/content/heart-failure-zigong/1.3/)

Steps to run the files:

```markdown
- Step 1: Run `pip install -r requirements.txt` for installing all the required packages.
- Step 2: Run `python main.py` in terminal [This is for cross-validation that finds the model with the highest recall among 10 times 10 folds]
- Step 3: Run `python final_refined_dataframe.py` in terminal [This is the chosen final XGBoost model with 17 features]
```