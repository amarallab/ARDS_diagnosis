import string
import json
import pickle
import re
from datetime import timedelta
import pandas as pd
import numpy as np
from nltk import word_tokenize
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier


def mark_hypoxemic_episodes(pf_ratio_table, peep, encounter_column):
    """
    For each patient in the PF ratio table, this function will flag hypoxemia entries.
    If PEEP table is specified, the function will not flag as hypoxemic those encounters
    in which PEEP was never >= 5 cm H2O.
    This function assumes that PF ratio timestamps from each patient encounter
    are sorted from earliest to latest.

    Inputs:
    - pf_ratio_table: pandas dataframe
    - peep: pandas dataframe
    - encounter_column: str

    Outputs:
    - pf_ratio_table: pandas dataframe
    - hypoxemia_df: pandas dataframe
    """

    if peep is not None:
        max_peep = (
            peep.groupby(encounter_column)["peep_value"] \
            .max() \
            .to_frame("max_peep_value") \
            .reset_index()
        )
        encounters_with_invalid_peep = list(
            max_peep.loc[max_peep["max_peep_value"] < 5, encounter_column]
        )
        pf_ratio_table["hypoxemia"] = (pf_ratio_table["pf_ratio_value"] <= 300) & (
            ~pf_ratio_table[encounter_column].isin(encounters_with_invalid_peep)
        )
        
    else:
        # If no PEEP specified, then we assume all patients had PEEP >= 5 cm H2O.
        # It should be checked.
        pf_ratio_table["hypoxemia"] = pf_ratio_table["pf_ratio_value"] <= 300

    hypoxemia_df = pf_ratio_table.loc[pf_ratio_table["hypoxemia"]].copy()

    return pf_ratio_table, hypoxemia_df


def tokenizer_better(text):
    """
    Tokenize text by replacing punctuation and numbers with spaces

    """

    punc_list = string.punctuation + "0123456789,"
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    # text = text.lower().translate(t)
    tokens = word_tokenize(text)

    return tokens


def mark_abnormal_cxr(
    cxr_table,
    label_col,
    train_data_path=None,
    train_col=None,
    use_youden=False,
):
    """
    This function parses each CXR report, and assigns a probability and prediction
    of a chest X-ray having evidence of bilateral infiltrates.
    It trains an XGBoost classifier to perform this classification task.
    ***It will attempt to find a pickle file for the model***
    ***Otherwise it will train one.***

    Inputs:
    - cxr_table: pandas dataframe
    - label_col: str, name of column containing the true label  
    - train_data_path: str, path to the annotated cxr table for training.
    - train_col: list of str, column names for text and labels in training data.
                It assumes first element to be X, or the features/samples,
                and the second element to be the label.
    - use_youden: bool, whether to use Youden's J statistic to binarize probabilities
    """

    # Join up all words in segmented reports
    # Target data
    cxr_table["seg_cxr_text"] = cxr_table["seg_cxr_text"].str.replace(
        r"'", "", regex=True
    )
    cxr_table["seg_cxr_text"] = cxr_table["seg_cxr_text"].str.replace(
        r"\[", "", regex=True
    )
    cxr_table["seg_cxr_text"] = cxr_table["seg_cxr_text"].str.replace(
        r"\]", "", regex=True
    )
    cxr_table["seg_cxr_text"] = cxr_table["seg_cxr_text"].str.replace(
        r",", "", regex=True
    )
    
    X_test = cxr_table["seg_cxr_text"].to_numpy()
    try:
        Y_test = cxr_table[label_col].to_numpy()
        can_validate = True
    except KeyError:
        can_validate = False
        print(
            "ATTENTION: This CXR table isn't annotated/scored. Can't compare to true label."
        )
        
    # Check that pickle files exist alongside this script
    try:
        
        with open("./src/mc1t1_cxr_vectorizer.pickle", "rb") as vectorizer_file:
            vect = pickle.load(vectorizer_file)
            
        if can_validate:
            with open("./src/mc1t1_cxr_xgboost_model_early_stopping.pickle", "rb") as model_file:
                XG_model = pickle.load(model_file)
                
        else:
            with open("./src/mc1t1_cxr_xgboost_model_no_early_stopping.pickle", "rb") as model_file:
                XG_model = pickle.load(model_file)
                
    except FileNotFoundError:
        
        train_dataset = pd.read_csv(train_data_path)

        # Training data
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r"'", "", regex=True
        )
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r"\[", "", regex=True
        )
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r"\]", "", regex=True
        )
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r",", "", regex=True
        )
        if "Unnamed: 0" in train_dataset.columns:
            train_dataset = train_dataset.drop(columns="Unnamed: 0")
            train_dataset = train_dataset.drop_duplicates()
        else:
            pass

        # "Splitting" into training and testing sets
        X_train = train_dataset[train_col[0]].to_numpy()
        Y_train = train_dataset[train_col[1]].to_numpy()

        # Vectorizing the "features"
        vect = CountVectorizer(
            tokenizer=tokenizer_better, ngram_range=(1, 2), max_features=200
        )
        vect.fit(X_train)
        X_train_vectorized = vect.transform(X_train).toarray()
        X_test_vectorized = vect.transform(X_test).toarray()

        # Getting tuned hyperparameters for XGBoost
        if "mc1t1" in str(train_data_path):
            with open(
                "../Development_notebooks/hyperparameters/XG_hyperparams_corpus_1.json",
                "r"
                ) as xg_file:
                xg_hyperparams = json.load(xg_file)

        elif "mc1t2" in str(train_data_path):
            with open(
                "../Development_notebooks/hyperparameters/XG_hyperparams_corpus_2.json",
                "r"
                ) as xg_file2:
                xg_hyperparams = json.load(xg_file2)
                    
        else:
            raise ValueError(
                "The training data path should contain either 'mc1t1' or 'mc1t2' to load the correct hyperparameters."
                )

        xg_hyperparams["n_estimators"] = int(xg_hyperparams["n_estimators"])
        xg_hyperparams["early_stopping_rounds"] = int(
            xg_hyperparams["early_stopping_rounds"]
            )
        xg_hyperparams["max_depth"] = int(xg_hyperparams["max_depth"])
        xg_hyperparams["base_score"] = float(xg_hyperparams["base_score"])
        xg_hyperparams["learning_rate"] = float(xg_hyperparams["learning_rate"])
        xg_hyperparams["gamma"] = float(xg_hyperparams["gamma"])
        xg_hyperparams["min_child_weight"] = float(xg_hyperparams["min_child_weight"])
        xg_hyperparams["max_delta_step"] = float(xg_hyperparams["max_delta_step"])
        xg_hyperparams["subsample"] = float(xg_hyperparams["subsample"])

        if can_validate:
            # Training XGBoost
            XG_model = XGBClassifier(
                **xg_hyperparams,
                tree_method="hist",
                n_jobs=-1,
                random_state=0
                )
            
            XG_model.fit(
                X_train_vectorized,
                Y_train,
                eval_set=[(X_test_vectorized, Y_test)],
                verbose=False
                )

        else:
            # Training XGBoost
            xg_hyperparams.pop("early_stopping_rounds")
            XG_model = XGBClassifier(
                **xg_hyperparams,
                tree_method="hist",
                n_jobs=-1,
                random_state=0
                )
            XG_model.fit(X_train_vectorized, Y_train, verbose=False)
            
    # Now, annotate CXRs
    X_test_vectorized = vect.transform(X_test).toarray()
    cxr_table["cxr_score_probability"] = XG_model.predict_proba(X_test_vectorized)[:, 1]
    
    # How to binarize probabilities (Youden's J or 50% threshold)
    if use_youden:
        fpr, tpr, thresholds = roc_curve(
            Y_test,
            XG_model.predict_proba(X_test_vectorized)[:, 1]
            )
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        print(f"Youden's J threshold: {best_thresh}")
        
        cxr_table["cxr_score_predicted"] = (
            XG_model.predict_proba(X_test_vectorized)[:, 1] >= best_thresh).astype(bool)
        
    else:
        cxr_table["cxr_score_predicted"] = XG_model.predict(X_test_vectorized).astype(bool)

    return cxr_table


def mark_cxr_within_48h_of_hypoxemia(
    hypoxemia_df, cxr_table, encounter_column, cxr_dt_column
):
    """
    For each patient in the CXR report table, this function will flag
    CXR reports that are within 48 h of a hypoxemic entry.

    Inputs:
    - hypoxemia_df: pandas dataframe
    - cxr_table: pandas dataframe
    - encounter_column: str

    Outputs:
    - cxr_table: pandas dataframe
    - hypox_abn_cxr_48h: pandas dataframe
    """

    bi_encounters = list(cxr_table[encounter_column].drop_duplicates())
    hypox_encounters = list(hypoxemia_df[encounter_column].drop_duplicates())
    cxr_table["within_48h"] = False

    for encounter_id in bi_encounters:
        if encounter_id not in hypox_encounters:
            continue

        cxr_encntr_mask = cxr_table[encounter_column] == encounter_id

        # Getting datetimes of hypoxemia
        hypox_encntr_mask = hypoxemia_df[encounter_column] == encounter_id
        hypoxemia_times = list(
            hypoxemia_df.loc[hypox_encntr_mask, "pf_ratio_timestamp"].drop_duplicates()
        )

        for timestamp in hypoxemia_times:
            within = abs(
                cxr_table.loc[cxr_encntr_mask, cxr_dt_column] - timestamp
            ) <= timedelta(hours=48)

            cxr_table.loc[cxr_encntr_mask, "within_48h"] = (
                within | cxr_table.loc[cxr_encntr_mask, "within_48h"]
            )

    cxr_table.loc[:, "hypox_and_pred_abn_cxr_48h"] = (
        cxr_table.loc[:, "cxr_score_predicted"] & cxr_table.loc[:, "within_48h"]
    )
    hypox_pred_abn_cxr_48h = cxr_table.loc[cxr_table["hypox_and_pred_abn_cxr_48h"]]

    return cxr_table, hypox_pred_abn_cxr_48h


def mark_cxr_within_48h_of_post_vent_hypoxemia(
    hypoxemia_df, cxr_table, encounter_column, cxr_dt_column
):
    """
    For each patient in the CXR report table, this function will flag
    CXR reports that are within 48 h of a hypoxemic entry that occurred
    after start of intubation.

    Inputs:
    - hypoxemia_df: pandas dataframe
    - cxr_table: pandas dataframe
    - encounter_column: str

    Outputs:
    - cxr_table: pandas dataframe
    - hypox_abn_cxr_48h: pandas dataframe
    """

    bi_encounters = list(cxr_table[encounter_column].drop_duplicates())
    post_intub = (
        hypoxemia_df["pf_ratio_timestamp"] >= hypoxemia_df["vent_start_timestamp"]
    )
    hypox_post_intubation = hypoxemia_df.loc[post_intub]
    hypox_post_intubation_encounters = list(
        hypox_post_intubation[encounter_column].drop_duplicates()
    )
    cxr_table["within_48h"] = False

    for encounter_id in bi_encounters:
        if encounter_id not in hypox_post_intubation_encounters:
            continue

        cxr_encntr_mask = cxr_table[encounter_column] == encounter_id

        # Getting datetimes of hypoxemia
        hypox_encntr_mask = hypox_post_intubation[encounter_column] == encounter_id
        hypoxemia_times = list(
            hypox_post_intubation.loc[
                hypox_encntr_mask, "pf_ratio_timestamp"
            ].drop_duplicates()
        )

        for timestamp in hypoxemia_times:
            within = abs(
                cxr_table.loc[cxr_encntr_mask, cxr_dt_column] - timestamp
            ) <= timedelta(hours=48)

            cxr_table.loc[cxr_encntr_mask, "within_48h"] = (
                within | cxr_table.loc[cxr_encntr_mask, "within_48h"]
            )

    cxr_table.loc[:, "hypox_and_pred_abn_cxr_48h"] = (
        cxr_table.loc[:, "cxr_score_predicted"] & cxr_table.loc[:, "within_48h"]
    )
    hypox_pred_abn_cxr_48h = cxr_table.loc[cxr_table["hypox_and_pred_abn_cxr_48h"]]

    return cxr_table, hypox_pred_abn_cxr_48h


def mark_note_within_7d(
    attn_notes, hypoxemia_df, hypox_pred_abn_cxr_48h, encounter_column, cxr_dt_column
):
    """
    Will add a column to attn_notes to flag whether the note is within
    7 days of the latest of PF ratio <= 300 mm Hg or abnormal CXR.

    Inputs:
    - attn_notes: pandas dataframe, attending physician notes data
    - hypoxemia_df: pandas dataframe, encounters with hypoxemia
    - hypox_pred_abn_cxr_48h: pandas dataframe
    - encounter_column: str

    Outputs:
    - attn_notes: pandas dataframe, with a flag column
    """

    # Get the latest of hypoxemia or abnormal CXR instances
    latest = pd.merge(
        hypoxemia_df[[encounter_column, "pf_ratio_timestamp"]],
        hypox_pred_abn_cxr_48h[[encounter_column, cxr_dt_column]],
        how="right",
    )

    f = abs(latest["pf_ratio_timestamp"] - latest[cxr_dt_column]) <= pd.Timedelta(
        hours=48
    )
    latest = latest.loc[f]
    latest["latest_time"] = latest[["pf_ratio_timestamp", cxr_dt_column]].max(axis=1)
    latest = latest[[encounter_column, "latest_time"]].drop_duplicates()

    # Now do the flagging
    notes_encounters = list(attn_notes[encounter_column].drop_duplicates())
    hypox_abn_cxr_encounters = list(latest[encounter_column].drop_duplicates())
    attn_notes["within_7d"] = False

    for encounter_id in notes_encounters:
        if encounter_id not in hypox_abn_cxr_encounters:
            continue

        notes_encntr_mask = attn_notes[encounter_column] == encounter_id

        # Getting latest datetimes of hypoxemia or abnormal CXR
        hypox_abn_cxr_encntr_mask = latest[encounter_column] == encounter_id
        latest_times = list(
            latest.loc[hypox_abn_cxr_encntr_mask, "latest_time"].drop_duplicates()
        )

        for timestamp in latest_times:
            flag1 = (
                attn_notes.loc[notes_encntr_mask, "notes_timestamp"] - timestamp
            ) >= timedelta(days=-1)
            flag2 = (
                attn_notes.loc[notes_encntr_mask, "notes_timestamp"] - timestamp
            ) <= timedelta(days=7)
            within = flag1 & flag2

            attn_notes.loc[notes_encntr_mask, "within_7d"] = (
                within | attn_notes.loc[notes_encntr_mask, "within_7d"]
            )

    return attn_notes


def mark_notes_with_ml(attn_notes, train_data_path, train_col=None):
    """
    This function adds a column to attending notes table with the prediction
    of pneumonia done by a XGBoost classifier trained to adjudicate pneumonia.
    This function is practically similar to mark_abnormal_cxr.
    ***It will attempt to find a pickle file for the model***
    ***Otherwise it will train one.***

    Inputs:
    - attn_notes: pandas dataframe
    - train_data_path: str, path to the training data.
    - train_col: list of str, column names for text and labels in training data.
                It assumes first element to be X, or the features/samples,
                and the second element to be the label.
    """

    # Join up all words in segmented reports
    # Test data
    attn_notes["seg_pneumonia"] = attn_notes["seg_pneumonia"].str.replace(
        r"'", "", regex=True
    )
    attn_notes["seg_pneumonia"] = attn_notes["seg_pneumonia"].str.replace(
        r"\[", "", regex=True
    )
    attn_notes["seg_pneumonia"] = attn_notes["seg_pneumonia"].str.replace(
        r"\]", "", regex=True
    )
    attn_notes["seg_pneumonia"] = attn_notes["seg_pneumonia"].str.replace(
        r",", "", regex=True
    )
    
    f = attn_notes[train_col[0]] != "Invalid"
    X_test = attn_notes.loc[f, train_col[0]].to_numpy()
    
    # Check that pickle files exist alongside this script
    try:
        
        with open("./src/mc1t1_notes_pna_vectorizer.pickle", "rb") as vectorizer_file:
            vect = pickle.load(vectorizer_file)
            
        with open("./src/mc1t1_notes_pna_xgboost_model.pickle", "rb") as model_file:
            XG_model = pickle.load(model_file)
                
    except FileNotFoundError:
        
        train_dataset = pd.read_csv(train_data_path / "pneumonia_attending_notes.csv")

        # Training data
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r"'", "", regex=True
        )
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r"\[", "", regex=True
        )
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r"\]", "", regex=True
        )
        train_dataset[train_col[0]] = train_dataset[train_col[0]].str.replace(
            r",", "", regex=True
        )
        if "Unnamed: 0" in train_dataset.columns:
            train_dataset = train_dataset.drop(columns="Unnamed: 0")
            train_dataset = train_dataset.drop_duplicates()
        else:
            pass

        # "Splitting" into training and testing sets
        X_train = train_dataset[train_col[0]].to_numpy()
        Y_train = train_dataset[train_col[1]].to_numpy()

        # Vectorizing the "features"
        vect = CountVectorizer(
            tokenizer=tokenizer_better, ngram_range=(1, 2), max_features=200
        )
        vect.fit(X_train)
        X_train_vectorized = vect.transform(X_train).toarray()
        X_test_vectorized = vect.transform(X_test).toarray()

        # Training XGBoost
        with open(
            "../Development_notebooks/hyperparameters/pna_XG_hyperparams.json",
            "r"
            ) as pna_file:
            hyperparams = json.load(pna_file)

        hyperparams["n_estimators"] = int(hyperparams["n_estimators"])
        hyperparams["max_depth"] = int(hyperparams["max_depth"])
        hyperparams["base_score"] = float(hyperparams["base_score"])
        hyperparams["learning_rate"] = float(hyperparams["learning_rate"])
        hyperparams["gamma"] = float(hyperparams["gamma"])
        hyperparams["min_child_weight"] = float(hyperparams["min_child_weight"])
        hyperparams["max_delta_step"] = float(hyperparams["max_delta_step"])
        hyperparams["subsample"] = float(hyperparams["subsample"])

        XG_model = XGBClassifier(
            **hyperparams, tree_method="hist", n_jobs=-1, random_state=0
        )
        XG_model.fit(X_train_vectorized, Y_train)

    # Now, annotate
    X_test_vectorized = vect.transform(X_test).toarray()
    attn_notes.loc[f, "pneumonia_probability"] = XG_model.predict_proba(
        X_test_vectorized
    )[:, 1]
    attn_notes.loc[f, "pneumonia_predicted"] = XG_model.predict(
        X_test_vectorized
    ).astype(bool)

    return attn_notes


def text_match_risk_factors(attn_notes):
    """
    Adds boolean columns (flags) for specific risk factors
    and for cardiogenic language. These flags represent whether
    the note text contained the regular expression pattern.

    Inputs:
    - attn_notes: pandas dataframe, attending physician notes data

    Outputs:
    - attn_notes: pandas dataframe, with a flag columns added
    """

    # patterns to search for
    sepsis_exclusion = [
        "r/o sepsis",
        "no longer in",
        "sepsis or cardiogenic shock",
        "cardiogenic vs septic",
        "potential for septic shock",
        "cardiac vs septic",
        "searching for evidence of",
        "shock mixed cardiogenic/vasodilatory",
        "sepsis vs cardiogemnic",
        "shock-septic vs cargdiogenic",
        "shock-septic vs cardiogenic",
        "severe sepsis resolved",
        "now off low dose vasopressor",
        "shock-septic vs hypovolemic resolved",
        "septic schock-resolved",
        "shock-septic vs hypovolemic",
        "septic shock off pressors",
        "cannot rule out septic/vasodilatory shock",
        "previously vasoactive support for sepsis",
        "billiary sepsis",
        "septic shock secondary to esbl bacteremia",
        "c/b septic joints",
        "septic shock due to pseudomonas bacteremia",
        "no evidence of hemorrhage or sepsis",
        "no evidence of ongoing hemorrhage or sepsis",
        "admitted with septic shock about months ago",
        "history of aspergillus pneumonia/sepsis",
        "mssa bactermia septic shock resolved",
        "hypotension/sepsis vs hypercoaguable",
        "septic shock ards copd exacerbation hcap bacteremia",
        "septic shock found to have klebsiella bacteremia",
        "suspected sepsis",
        "septic emboli syndrome",
        "takotsubo possible",
        "takatsubo with possible",
        "septic shock suspect recurrent takatsubo's",
        "without active hemorrhage or sepsis",
        "w/u for sepsis underway",
        "also concern for sepsis",
        "cytopenias likely due to sepsis",
        "sedation sepsis",
        "septic shock with picture",
    ]
    shock_exclusion = [
        "is no longer in septic shock",
        "chest compressions or shocks",
        "shock now resolved",
        "potential for septic",
        "septic shock off pressors",
        "septic shock with picture",
        "no longer in shock",
        "septic shock-resolved",
        "septic shock due to pseudomonas bacteremia",
        "shock has resolved",
        "septic shock source uncertain",
        "not in shock",
        "weaned off pressors",
        "septic shock due to e-coli bactermia resolved",
        "most likely distributive liver failure vs sepsis",
        "shock--improving",
        "terminated by shock",
        "icd interrogation reported two shocks",
        "unlikely to be cardiogenic shock",
        "cpr/shocks",
        "shocks before rosc",
        "underwent cardiopulmonary resuscitation and",
        "now off low dose vasopressor requirement",
        "schock-resolved",
        "vs hypovolemic resolved",
        "cannot rule out septic/vasodilatory",
        "obstructive shock due to pe and septic",
        "no operative intervention recommended except in shock situation",
    ]
    cardiac_arrest_exclusion = [
        "was likely imminent",
        "infection/shock/arrest",
        "cva",
        "near respiratory arrest",
        "lack of history of prolonged arrest",
        "pea arrest w rosc approx",
        "setting of severe",
        "s/p respiratory arrest yesterday",
        "prior to cardiopulmonary arrest",
        "impending respiratory arrest",
        "aortic dissection repair with respiratory arrest requiring reintubation",
        "cardiopulmonary arrest-neurologic status appears to be at baseline or improved",
        "pea arrest now intubated and in septic shock",
        "tca ingestion",
        "ingestion of tca",
        "post cardiopulmonary arrest-no neurologic deficit compared to baseline",
        "c/b septic shock/arrest with prior intubation",
        "acute respiratory arrest",
        "circulatory arrest",
        "circ arrest",
        "s/p vf arrest",
    ]
    pulm_contusion_pattern = r"(?i)(?<!\w)pulmonary (?:contusion|contusions)(?!\w)"
    vasculitis_pattern = r"(?i)(?<!\w)(?:pulmonary vasculitis|vasculitis)(?!\w)"
    drowning_pattern = r"(?i)(?<!\w)(?:drowned|drowning)(?!\w)"
    overdose_pattern = r"(?i)(?<!\w)(?:overdose|drug overdose)(?!\w)"
    cardiogenic_pattern = r"(?i)(?<!\w)(?<!non\s)(?<!non-)(?:cardiogenic|cardigenic|cardiogemic|cardiac pulmonary edema|cardiac and septic shock|Shock.{1,15}suspect.{1,15}RV failure)(?!\w)"
    non_cardiogenic_pattern = r"(?i)(?<!\w)(?:non(?:-|\s)cardiogenic|noncardiogenic|non(?:-|\s)cardigenic|noncardigenic)(?!\w)"

    # Adding the flag/boolean columns.
    # Sepsis is a little special
    boolean_sepsis_list = []
    for seg_sepsis in attn_notes["seg_sepsis"]:
        # If any of the exclusion phrases is found in the text snippet, make False.
        # Otherwise, make True.
        boolean_sepsis_list.append(
            not any([phrase in seg_sepsis for phrase in sepsis_exclusion])
            and seg_sepsis != "Invalid"
        )
    attn_notes["sepsis_predicted"] = boolean_sepsis_list

    boolean_shock_list = []
    for seg_shock in attn_notes["seg_shock"]:
        # If any of the exclusion phrases is found in the text snippet, make False.
        # Otherwise, make True.
        boolean_shock_list.append(
            not any([phrase in seg_shock for phrase in shock_exclusion])
            and seg_shock != "Invalid"
        )
    attn_notes["shock_predicted"] = boolean_shock_list

    # Also cardiac arrest is special
    boolean_arrest_list = []
    for seg_cardiac_arrest in attn_notes["seg_cardiac_arrest"]:
        # If any of the exclusion phrases is found in the text snippet, make False.
        # Otherwise, make True.
        boolean_arrest_list.append(
            not any(
                [phrase in seg_cardiac_arrest for phrase in cardiac_arrest_exclusion]
            )
            and seg_cardiac_arrest != "Invalid"
        )
    attn_notes["cardiac_arrest_predicted"] = boolean_arrest_list

    attn_notes["pulmonary_contusion_predicted"] = attn_notes.notes_text.str.contains(
        pulm_contusion_pattern
    )
    attn_notes["vasculitis_predicted"] = attn_notes.notes_text.str.contains(
        vasculitis_pattern
    )
    attn_notes["drowning_predicted"] = attn_notes.notes_text.str.contains(
        drowning_pattern
    )
    attn_notes["overdose_predicted"] = attn_notes.notes_text.str.contains(
        overdose_pattern
    )
    attn_notes["cardiogenic_predicted"] = attn_notes.notes_text.str.contains(
        cardiogenic_pattern
    )
    attn_notes["non_cardiogenic_predicted"] = attn_notes.notes_text.str.contains(
        non_cardiogenic_pattern
    )

    # Any risk factor identified?
    attn_notes["risk_factor_identified"] = (
        attn_notes["pneumonia_predicted"].fillna(False) \
        | attn_notes["sepsis_predicted"].fillna(False) \
        | attn_notes["vasculitis_predicted"].fillna(False) \
        | attn_notes["drowning_predicted"].fillna(False) \
        | attn_notes["overdose_predicted"].fillna(False) \
        | attn_notes["pulmonary_contusion_predicted"].fillna(False) \
        | (attn_notes["non_cardiogenic_predicted"] & attn_notes["shock_predicted"])
    )

    return attn_notes


def diagnose_or_exclude_encounters(
    attn_notes, hypox_pred_abn_cxr_48h, encounter_column
):
    """
    Reads previously labelled attending notes table, and sorts its
    encounters by whether they got diagnosed, excluded due to only
    having cardiogenic source of bilateral infiltrates, or no
    determination, requiring objective cardiac failure assessment.

    Inputs:
    - attn_notes: pandas dataframe, attending physician notes data
    - hypox_pred_abn_cxr_48h: pandas dataframe, encounters with hypoxemia
                                                and abnormal CXRs within 48 h
    - encounter_column: str, name of column containing the relevant encounter ID

    Outputs:
    - attn_notes: pandas dataframe, with a flag column
    - diagnosed: pandas dataframe, containing diagnosed encounters
    - discarded: pandas dataframe, containing encounters having exclusive cardiogenic explanation
    - objective_assessment: pandas dataframe, containing indeterminate encounters
    """

    # This is to include encounters that had qualified hypoxemia, but do not appear in notes file
    # These should end up going through cardiac failure objective assessment
    encounters_with_hypox_sans_notes = pd.merge(
        hypox_pred_abn_cxr_48h["encounter_id"], attn_notes, how="outer", indicator=True
    )
    mask = encounters_with_hypox_sans_notes["_merge"] == "left_only"
    encounters_with_hypox_sans_notes = encounters_with_hypox_sans_notes.loc[mask]

    # Table fulfilling Berlin criteria (Hypoxemia, bilateral infiltrates,
    # and risk factor within appropriate time windows)
    f = attn_notes["within_7d"] & attn_notes["risk_factor_identified"]
    diagnosed = attn_notes.loc[f]

    diagnosed_encounters = list(diagnosed[encounter_column].drop_duplicates())

    # Getting remaining encounters in the notes
    notes_sans_diagnosed = attn_notes.loc[
        ~attn_notes[encounter_column].isin(diagnosed_encounters)
    ]

    # Table with discarded cases: Try to get any notes where cardiac failure language was used
    g = (
        notes_sans_diagnosed["cardiogenic_predicted"]
        & notes_sans_diagnosed["shock_predicted"]
    ) | notes_sans_diagnosed["cardiac_arrest_predicted"]

    # Now, get notes where no risk factor could be identified within 7 days of qualified hypoxemia
    h = (
        ~notes_sans_diagnosed["risk_factor_identified"]
        & notes_sans_diagnosed["within_7d"]
    )
    discarded = notes_sans_diagnosed.loc[g & h]

    discarded_encounters = list(discarded[encounter_column].drop_duplicates())

    # Table with ambiguous cases that should go for objective cardiac failure assessment
    j = attn_notes["within_7d"] & ~notes_sans_diagnosed[encounter_column].isin(
        discarded_encounters
    )
    objective_assessment = notes_sans_diagnosed.loc[j]

    # Including potentially left-out encounters
    objective_assessment = pd.merge(
        encounters_with_hypox_sans_notes, objective_assessment, how="outer"
    )

    return attn_notes, diagnosed, discarded, objective_assessment


def find_matches(prefix, suffix, text):
    """
    Find values/statements of interest in ECHO reports
    through particular regex patterns. It will return
    the most frequent number or relevant text (both as str)
    in text, assuming that the regex pattern correctly
    specifies the capturing group.

    Input:
        - prefix: dict, containing first part of regex pattern
        - suffix: dict, containing second part of regex pattern
        - text: str, text from ECHO reports to parse

    Return:
        - Most frequent match found. If no match, return None
    """

    m = re.findall(prefix + suffix, text)

    if len(m) > 0:
        # This is essentially getting the mode
        return max(set(m), key=m.count)

    return None


def flag_echos(echo_reports, prefix, suffix):
    """
    Add columns to flag the presence of values/statements
    of interest in ECHO reports according to the criteria
    laid out in the objective assessment of cardiac failure.
    It uses regex patterns specified in prefix and suffix to
    parse ECHO reports for these criteria, using the keys to
    name the columns in pandas dataframes, and populating the
    columns with numeric values (if criterion is numeric), or
    string (if simple presence/absence of criterion).

    Inputs:
        - echo_reports - pandas dataframe
        - prefix - dict, regex flags/prefixes
        - suffix - dict, regex flags/suffixes

    Returns:
        - echo_reports - pandas dataframe
    """

    for criterion in prefix.keys():
        flag_column = []
        value_column = []
        none_flag_count = 0
        none_value_count = 0
        prefix_list = prefix[criterion]
        sufijo = suffix[criterion]

        for i in echo_reports["echo_text"]:
            temp_value = []
            temp_flag = []

            for j in prefix_list:
                try:
                    temp_value.append(find_matches(j, sufijo, i))
                except TypeError:
                    temp_value.append(None)

            for j in prefix_list:
                try:
                    temp_flag.append(find_matches(j, "", i))
                except TypeError:
                    temp_flag.append(None)

            # If the temporary list got filled with just None
            if temp_flag.count(None) == len(temp_flag):
                none_flag_count += 1
                flag_column.append(0)
            else:
                flag_column.append(1)

            if temp_value.count(None) == len(temp_value):
                none_value_count += 1
                value_column.append(None)
                continue

            temp_sans_none = [x for x in temp_value if x is not None]

            if "ejection" in criterion or "ef" in criterion:

                temp_value_store = []

                for value in temp_sans_none:
                    if "-" in value:
                        split_value = value.split("-")
                        temp_value_store.append(
                            float(split_value[1].strip())
                        )  # This picks the upper bound
                    else:
                        temp_value_store.append(float(value))

                value_column.append(max(temp_value_store))

            elif (
                "bypass" in criterion
                or "hypertrophy" in criterion
                or "dysfunction" in criterion
            ):
                # Assuming the value we want from each text would be repeated the most
                mode = max(set(temp_sans_none), key=temp_sans_none.count)

                # Sometimes phrases are cut by a newline character '\n'
                if "\n" in mode:
                    split_mode = mode.split("\n")
                    split_mode[-1] = split_mode[-1].strip()
                    value_column.append("".join(split_mode))
                else:
                    value_column.append(mode)

            elif "dimension" in criterion or "volume" in criterion:
                # Assuming the value we want from each text would be repeated the most
                mode = max(set(temp_sans_none), key=temp_sans_none.count)

                # This is to deal with numbers like '3. 7\n\n'
                if "." in mode:
                    split_mode = mode.split(" ")
                    value_column.append(float("".join(split_mode).strip("\n")))
                else:
                    value_column.append(float(mode))

            else:
                # Assuming the value we want from each text would be repeated the most
                mode = max(set(temp_sans_none), key=temp_sans_none.count)

        print(
            f"For {criterion}, there were {none_flag_count} entries not flagged and {none_value_count} null value matches.\n"
        )
        echo_reports[f"{criterion}_value"] = value_column
        echo_reports[f"{criterion}_flag"] = flag_column

    return echo_reports