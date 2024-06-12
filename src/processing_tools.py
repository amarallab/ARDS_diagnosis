import os
import json
import pandas as pd
import numpy as np


############################ File I/O #############################
def _use_system_path_separator(x):
    """ "
    Replaces back and forward slashes with file separator used on current
    operating system.
    """

    x = x.replace("\\", os.path.sep)
    x = x.replace("/", os.path.sep)

    return x


def _data_location_pointer():
    """
    Obtains a dictionary of paths to datasets from a specific settings file.
    This settings file needs to be a UTF-8 formatted CSV file,
    should be at the same filesystem level as the file calling this function,
    and it should be named 'settings'.

    Output:
    settings       dict
    """

    path_to_settings = os.getcwd()

    if not os.path.exists(path_to_settings):
        raise EnvironmentError(
            rf"""
            Could not find directory reserved for settings:
            {path_to_settings}
        """
        )

    path_to_settings = os.path.join(path_to_settings, "settings.csv")

    if not os.path.exists(path_to_settings):
        raise EnvironmentError(
            rf"""
            Could not find settings.csv file:
            {path_to_settings}

            This file needs to be UTF-8 formatted
            csv file with two columns: key, value
        """
        )

    settings = pd.read_csv(path_to_settings)

    if not all(settings.columns == ["key", "value"]):
        raise EnvironmentError(
            rf"""
            settings.csv must have exactly two different
            columns, namely key and value.
            
            {path_to_settings}
        """
        )

    settings = settings.drop_duplicates()

    if any(settings["key"].duplicated()):
        raise EnvironmentError(
            rf"""
            At least one key within settings.csv is
            duplicated and therefore ambiguous
            
            {path_to_settings}
        """
        )

    settings = settings.set_index("key", verify_integrity=True)["value"].to_dict()

    return settings


def get_path(dataset=None, extension=None):
    """
    Returns subfolder containing hospital-specific data.

    Input:
        dataset     str, name of hospital dataset, e.g.: 'Northwestern Medicine'
        extension   str, optional, subfolder
    Output:
        path        str, folder
    """

    if extension is not None:
        extension = _use_system_path_separator(extension)

    settings = _data_location_pointer()

    if dataset.lower() in settings.keys():
        ref_folder = settings[dataset.lower()]

        if extension is not None:
            path = os.path.join(ref_folder, extension)
        else:
            path = ref_folder

        if os.path.exists(path):
            return path

        raise ValueError(
            f"""Function get_path could not get a currently-existing directory.
            Check that this path actually exists in your filesystem: {path}
            """
        )

    raise KeyError()


def _read_table(table, dataset=None, subfolder=None, columns=None):
    """
    Loads tables from a user specified path and set of columns.
    It attemps to support the same file extensions as pandas' read_excel:
    `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt`.
    It can also support reading JSON and CSV files.

    Input:
    table        Name of table - str
    dataset      Name of hospital dataset, e.g.: 'Northwestern Medicine' - str
    subfolder    Specific subfolder within dataset folder - str
    columns      Names of important columns in the table - list

    Output:
    table     pandas dataframe
    """

    if dataset is not None:
        prefix_path = get_path(dataset, subfolder)
        p = os.path.join(prefix_path, table)

        if ".csv" in table:
            df = pd.read_csv(p, usecols=columns)
            return df
        if ".xls" in table or ".od" in table:
            df = pd.read_excel(p, usecols=columns)
            return df
        if ".json" in table:
            with open(p, "r") as file:
                raw_json = json.load(file)
            df = pd.DataFrame(raw_json, columns=columns)
            return df

        raise ValueError(
            f"""
            This is your input for table: {table}. It should contain file extension.
            However, it can only be csv, Excel, JSON files.
            """
        )

    else:
        raise ValueError("Must specify a dataset or hospital name")


def read_in_files():
    """
    This is the main file I/O routine for stitching datasets for ARDS diagnosis.
    It prompts the user to specify which dataset to read.
    The function will read in a specs file that should be in the root directory
    of the dataset in question. This file consists of key/value pairs which specify
    file names, columns of interest, and whether files are within subfolders.

    returns - all relevant tables, as pandas dataframes
            - specs, as dict
    """

    print(
        """
    Welcome to the dataset input prompts!\n
    For yes/no questions, make sure to write yes or no (capitalization doesn't matter)
    But you can also leave blank if wanting to answer 'no'.
    """
    )

    while True:
        try:
            dataset = input("Which dataset do you want to read in?  ")
            get_path(dataset=dataset)
        except KeyError:
            print(
                """
                You're using a hospital name not present in the settings.csv file.
                Try again with a hospital name string that exists in the settings.csv file
                """
            )
            continue
        else:
            break

    print("\n")

    print(
        """
        Time to load files pertinent to ARDS diagnosis! These include:
        - Dictionary table (if available or specified separately in your dataset)
        - PF ratios
        - PEEP (if available or specified separately in your dataset)
        - Chest X-ray (CXR) reports
        - Attending physician notes
        - Echocardiography reports
        - Brain/Beta Natriuretic Peptide lab values\n
        """
    )

    dummy_wait = input("Done reading? Press Enter to continue")

    print(
        """
        The above tables will be loaded from a "specs" file that should be in the root directory of your dataset.
        This specs file should contain key/value pairs specifying the following for each of the tables mentioned above:
        - File name, with file extension (e.g. filename.csv or filename.xlsx)

        - If the particular table is located within a subfolder, the name of the subfolder

        - A list of the columns that are relevant. Specifically, you'd need columns detailing:
            + Patient and/or hospitalization identifiers
            + Admission/discharge dates (if available in such table); also, ICU start/end dates, if available.
            + Date/datetime of the report or lab value.
            + The actual value or report of interest.
            + If the table is text-based, the score/label/annotation (if available).
        
        NOTES:
        When entering multiple column names in a single value field, separate them with a single whitespace.
        If a table has more than one possible timestamp for a value (e.g., order and procedure datetimes for a report), specify both.
        """
    )

    dummy_wait = input("When you're ready, press Enter to proceed to file reading")

    spec_file = _read_table("specs.csv", dataset=dataset)

    if not all(spec_file.columns == ["key", "value"]):
        raise EnvironmentError(
            r"""
            specs.csv must have exactly two different
            columns, namely key and value.
            """
        )

    spec_file = spec_file.drop_duplicates()

    if any(spec_file["key"].duplicated()):
        raise EnvironmentError(
            r"""
            At least one key within specs.csv is
            duplicated and therefore ambiguous
            """
        )

    specs = spec_file.set_index("key", verify_integrity=True)["value"].to_dict()

    print("\n")
    ### Dictionary table, if specified or needed ###
    if isinstance(specs["dict_filename"], str):
        dict_patient_id_col = (
            []
            if specs["dict_patient_id_col"] is np.nan
            else specs["dict_patient_id_col"].split()
        )
        dict_encounter_id_col = (
            []
            if specs["dict_encounter_id_col"] is np.nan
            else specs["dict_encounter_id_col"].split()
        )
        dict_icu_id_col = (
            []
            if specs["dict_icu_id_col"] is np.nan
            else specs["dict_icu_id_col"].split()
        )
        dict_id_cols = dict_patient_id_col + dict_encounter_id_col + dict_icu_id_col
        dict_hosp_icu_time_cols = (
            []
            if specs["dict_hosp_and_or_icu_timestamp_cols"] is np.nan
            else specs["dict_hosp_and_or_icu_timestamp_cols"].split()
        )
        dict_columns = dict_id_cols + dict_hosp_icu_time_cols

        dictionary = _read_table(
            specs["dict_filename"],
            dataset=dataset,
            subfolder=(
                None if specs["dict_subfolder"] is np.nan else specs["dict_subfolder"]
            ),
            columns=dict_columns,
        )

        print(f"{specs['dict_filename']} read successfully!\n")

    else:
        dictionary = None
        print(
            "No dictionary table specified in specs file. You will need to create one manually."
        )

    ### PF ratio ###
    pf_patient_id_col = (
        []
        if specs["pf_ratio_patient_id_col"] is np.nan
        else specs["pf_ratio_patient_id_col"].split()
    )
    pf_encounter_id_col = (
        []
        if specs["pf_ratio_encounter_id_col"] is np.nan
        else specs["pf_ratio_encounter_id_col"].split()
    )
    pf_icu_id_col = (
        []
        if specs["pf_ratio_icu_id_col"] is np.nan
        else specs["pf_ratio_icu_id_col"].split()
    )
    pf_id_cols = pf_patient_id_col + pf_encounter_id_col + pf_icu_id_col
    pf_hosp_icu_time_cols = (
        []
        if specs["pf_ratio_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["pf_ratio_hosp_and_or_icu_timestamp_cols"].split()
    )
    pf_vent_time_cols = (
        []
        if specs["pf_ratio_vent_timestamp_col"] is np.nan
        else specs["pf_ratio_vent_timestamp_col"].split()
    )
    pf_value_time_cols = specs["pf_ratio_value_timestamp_cols"].split()
    pf_time_cols = pf_hosp_icu_time_cols + pf_vent_time_cols + pf_value_time_cols
    pf_value_cols = specs["pf_ratio_value_col"].split()
    pf_columns = pf_id_cols + pf_time_cols + pf_value_cols

    pf = _read_table(
        specs["pf_ratio_filename"],
        dataset=dataset,
        subfolder=(
            None
            if specs["pf_ratio_subfolder"] is np.nan
            else specs["pf_ratio_subfolder"]
        ),
        columns=pf_columns,
    )

    pf[pf_time_cols] = pf[pf_time_cols].apply(pd.to_datetime, errors="coerce")

    print(f"{specs['pf_ratio_filename']} read successfully!\n")

    ### PEEP, if needed ###
    if isinstance(specs["peep_filename"], str):
        peep_patient_id_col = (
            []
            if specs["peep_patient_id_col"] is np.nan
            else specs["peep_patient_id_col"].split()
        )
        peep_encounter_id_col = (
            []
            if specs["peep_encounter_id_col"] is np.nan
            else specs["peep_encounter_id_col"].split()
        )
        peep_icu_id_col = (
            []
            if specs["peep_icu_id_col"] is np.nan
            else specs["peep_icu_id_col"].split()
        )
        peep_id_cols = peep_patient_id_col + peep_encounter_id_col + peep_icu_id_col
        peep_hosp_icu_time_cols = (
            []
            if specs["peep_hosp_and_or_icu_timestamp_cols"] is np.nan
            else specs["peep_hosp_and_or_icu_timestamp_cols"].split()
        )
        peep_value_time_cols = specs["peep_value_timestamp_cols"].split()
        peep_time_cols = peep_hosp_icu_time_cols + peep_value_time_cols
        peep_value_cols = specs["peep_value_col"].split()
        peep_columns = peep_id_cols + peep_time_cols + peep_value_cols

        peep = _read_table(
            specs["peep_filename"],
            dataset=dataset,
            subfolder=(
                None if specs["peep_subfolder"] is np.nan else specs["peep_subfolder"]
            ),
            columns=peep_columns,
        )

        peep[peep_time_cols] = peep[peep_time_cols].apply(
            pd.to_datetime, errors="coerce"
        )

        print(f"{specs['peep_filename']} read successfully!\n")

    else:
        peep = None
        print(
            "No PEEP table specified in specs file. File-reading code assumed there isn't one in this dataset."
        )

    ### Chest X-rays ###
    cxr_patient_id_col = (
        []
        if specs["cxr_patient_id_col"] is np.nan
        else specs["cxr_patient_id_col"].split()
    )
    cxr_encounter_id_col = (
        []
        if specs["cxr_encounter_id_col"] is np.nan
        else specs["cxr_encounter_id_col"].split()
    )
    cxr_icu_id_col = (
        [] if specs["cxr_icu_id_col"] is np.nan else specs["cxr_icu_id_col"].split()
    )
    cxr_id_cols = cxr_patient_id_col + cxr_encounter_id_col + cxr_icu_id_col
    cxr_hosp_icu_time_cols = (
        []
        if specs["cxr_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["cxr_hosp_and_or_icu_timestamp_cols"].split()
    )
    cxr_value_time_cols = specs["cxr_value_timestamp_cols"].split()
    cxr_time_cols = cxr_hosp_icu_time_cols + cxr_value_time_cols
    cxr_value_cols = specs["cxr_value_col"].split()
    cxr_label_cols = (
        []
        if specs["cxr_value_label_cols"] is np.nan
        else specs["cxr_value_label_cols"].split()
    )
    cxr_columns = cxr_id_cols + cxr_time_cols + cxr_value_cols + cxr_label_cols

    bi = _read_table(
        specs["cxr_filename"],
        dataset=dataset,
        subfolder=None if specs["cxr_subfolder"] is np.nan else specs["cxr_subfolder"],
        columns=cxr_columns,
    )

    bi[cxr_time_cols] = bi[cxr_time_cols].apply(pd.to_datetime, errors="coerce")

    print(f"{specs['cxr_filename']} read successfully!\n")

    ### Attending physician notes ###
    notes_patient_id_col = (
        []
        if specs["notes_patient_id_col"] is np.nan
        else specs["notes_patient_id_col"].split()
    )
    notes_encounter_id_col = (
        []
        if specs["notes_encounter_id_col"] is np.nan
        else specs["notes_encounter_id_col"].split()
    )
    notes_icu_id_col = (
        [] if specs["notes_icu_id_col"] is np.nan else specs["notes_icu_id_col"].split()
    )
    notes_id_cols = notes_patient_id_col + notes_encounter_id_col + notes_icu_id_col
    notes_hosp_icu_time_cols = (
        []
        if specs["notes_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["notes_hosp_and_or_icu_timestamp_cols"].split()
    )
    notes_value_time_cols = specs["notes_value_timestamp_cols"].split()
    notes_time_cols = notes_hosp_icu_time_cols + notes_value_time_cols
    notes_value_cols = specs["notes_value_col"].split()
    notes_label_cols = (
        []
        if specs["notes_value_label_cols"] is np.nan
        else specs["notes_value_label_cols"].split()
    )
    notes_columns = (
        notes_id_cols + notes_time_cols + notes_value_cols + notes_label_cols
    )

    notes = _read_table(
        specs["notes_filename"],
        dataset=dataset,
        subfolder=(
            None if specs["notes_subfolder"] is np.nan else specs["notes_subfolder"]
        ),
        columns=notes_columns,
    )

    notes[notes_time_cols] = notes[notes_time_cols].apply(
        pd.to_datetime, errors="coerce"
    )

    print(f"{specs['notes_filename']} read successfully!\n")

    ### Echocardiography reports ###
    echo_patient_id_col = (
        []
        if specs["echo_patient_id_col"] is np.nan
        else specs["echo_patient_id_col"].split()
    )
    echo_encounter_id_col = (
        []
        if specs["echo_encounter_id_col"] is np.nan
        else specs["echo_encounter_id_col"].split()
    )
    echo_icu_id_col = (
        [] if specs["echo_icu_id_col"] is np.nan else specs["echo_icu_id_col"].split()
    )
    echo_id_cols = echo_patient_id_col + echo_encounter_id_col + echo_icu_id_col
    echo_hosp_icu_time_cols = (
        []
        if specs["echo_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["echo_hosp_and_or_icu_timestamp_cols"].split()
    )
    echo_value_time_cols = specs["echo_value_timestamp_cols"].split()
    echo_time_cols = echo_hosp_icu_time_cols + echo_value_time_cols
    echo_value_cols = specs["echo_value_col"].split()
    echo_label_cols = (
        []
        if specs["echo_value_label_cols"] is np.nan
        else specs["echo_value_label_cols"].split()
    )
    echo_columns = echo_id_cols + echo_time_cols + echo_value_cols + echo_label_cols

    echo = _read_table(
        specs["echo_filename"],
        dataset=dataset,
        subfolder=(
            None if specs["echo_subfolder"] is np.nan else specs["echo_subfolder"]
        ),
        columns=echo_columns,
    )

    echo[echo_time_cols] = echo[echo_time_cols].apply(pd.to_datetime, errors="coerce")

    print(f"{specs['echo_filename']} read successfully!\n")

    ### Brain/Beta Natriuretic Peptide values ###
    bnp_patient_id_col = (
        []
        if specs["bnp_patient_id_col"] is np.nan
        else specs["bnp_patient_id_col"].split()
    )
    bnp_encounter_id_col = (
        []
        if specs["bnp_encounter_id_col"] is np.nan
        else specs["bnp_encounter_id_col"].split()
    )
    bnp_icu_id_col = (
        [] if specs["bnp_icu_id_col"] is np.nan else specs["bnp_icu_id_col"].split()
    )
    bnp_id_cols = bnp_patient_id_col + bnp_encounter_id_col + bnp_icu_id_col
    bnp_hosp_icu_time_cols = (
        []
        if specs["bnp_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["bnp_hosp_and_or_icu_timestamp_cols"].split()
    )
    bnp_value_time_cols = specs["bnp_value_timestamp_cols"].split()
    bnp_time_cols = bnp_hosp_icu_time_cols + bnp_value_time_cols
    bnp_value_cols = specs["bnp_value_col"].split()
    bnp_columns = bnp_id_cols + bnp_time_cols + bnp_value_cols

    bnp = _read_table(
        specs["bnp_filename"],
        dataset=dataset,
        subfolder=None if specs["bnp_subfolder"] is np.nan else specs["bnp_subfolder"],
        columns=bnp_columns,
    )

    bnp[bnp_time_cols] = bnp[bnp_time_cols].apply(pd.to_datetime, errors="coerce")

    print(f"{specs['bnp_filename']} read successfully!\n")

    return dictionary, pf, peep, bi, notes, echo, bnp, specs


################## Table pre-processing ####################################
def preprocess_tables(
    dictionary, pf_ratios, peep, cxr_reports, attn_notes, echo_reports, bnp_values
):
    """
    Function to perform general preprocessing routines:
    - Standardize value/report column names and corresponding timestamp columns
    - Drop duplicate rows
    - Drop rows where value/report, and corresponding timestamp, are both absent (or NULL)
    - If present, merge the dictionary file.
    - Sort by encounter or patient ID, and then by value/report timestamp

    This function will not rename encounter or patient IDs, and hospitalization/ICU dates,
    as these will be assumed to have the same column names accross tables.
    If that's not the case, manually correct for this.

    Returns:
    pf_ratios, peep, cxr_reports, attn_notes, echo, bnp_values - pandas dataframes
    """

    while True:
        try:
            dataset = input(
                "Again, what's your dataset? I have mild computer amnesia...  "
            )
            get_path(dataset=dataset)
        except KeyError:
            print(
                """
                You're using a hospital name not present in the settings.csv file.
                Try again with a hospital name string that exists in the settings.csv file
                """
            )
            continue
        else:
            break

    spec_file = _read_table("specs.csv", dataset=dataset)

    if not all(spec_file.columns == ["key", "value"]):
        raise EnvironmentError(
            r"""
            specs.csv must have exactly two different
            columns, namely key and value.
            """
        )

    spec_file = spec_file.drop_duplicates()

    if any(spec_file["key"].duplicated()):
        raise EnvironmentError(
            r"""
            At least one key within specs.csv is
            duplicated and therefore ambiguous
            """
        )

    specs = spec_file.set_index("key", verify_integrity=True)["value"].to_dict()

    ### Dictionary ###
    if dictionary is not None:
        print("\n\nDealing with dictionary table")
        dict_patient_id_col = (
            []
            if specs["dict_patient_id_col"] is np.nan
            else specs["dict_patient_id_col"].split()
        )
        dict_encounter_id_col = (
            []
            if specs["dict_encounter_id_col"] is np.nan
            else specs["dict_encounter_id_col"].split()
        )
        dict_icu_id_col = (
            []
            if specs["dict_icu_id_col"] is np.nan
            else specs["dict_icu_id_col"].split()
        )
        dict_hosp_icu_time_cols = (
            []
            if specs["dict_hosp_and_or_icu_timestamp_cols"] is np.nan
            else specs["dict_hosp_and_or_icu_timestamp_cols"].split()
        )
        dict_id_col = []

        # Converting time columns
        try:
            dictionary[dict_hosp_icu_time_cols] = dictionary[
                dict_hosp_icu_time_cols
            ].apply(pd.to_datetime)
            print("Columns with dates successfully converted to pd.datetime.")
        except TypeError:
            print(
                "Failed to convert columns that contain dates into datetime. Manually check."
            )

        # Renaming columns: IDs
        if len(dict_patient_id_col) == 1:
            dictionary = dictionary.rename(
                columns={dict_patient_id_col[0]: "patient_id"}
            )
            dict_id_col.append("patient_id")
            print("Patient ID column renamed successfully!")
        else:
            print("No patient ID column specified in dictionary table.")

        if len(dict_encounter_id_col) == 1:
            dictionary = dictionary.rename(
                columns={dict_encounter_id_col[0]: "encounter_id"}
            )
            dict_id_col.append("encounter_id")
            print("Hospitalization/Encounter ID column renamed successfully!")
        else:
            print(
                """
                No hospitalization/encounter ID column specified in dictionary table.
                How will this be useful for ARDS diagnosis, then?
                """
            )

        if len(dict_icu_id_col) == 1:
            dictionary = dictionary.rename(columns={dict_icu_id_col[0]: "icu_id"})
            dict_id_col.append("icu_id")
            print("ICU stay ID column renamed successfully!")
        else:
            print("No ICU stay ID column specified in dictionary table.")

        # Make column names lowercase
        dictionary.columns = dictionary.columns.str.lower()

        # Dropping duplicates and nulls (for this table, a null in any row is likely useless)
        dictionary = dictionary.drop_duplicates()
        dictionary = dictionary.dropna()

        # Sorting by IDs and then hospitalization/ICU timestamps
        dict_hosp_icu_time_cols = [
            value_time_col.lower() for value_time_col in dict_hosp_icu_time_cols
        ]
        dictionary = dictionary.sort_values(by=dict_id_col + dict_hosp_icu_time_cols)
        print("Successfully sorted table by IDs and then by timestamps.")

    else:
        pass

    ### PF ratios ###
    print("\nDealing with PF ratio table")
    pf_patient_id_col = (
        []
        if specs["pf_ratio_patient_id_col"] is np.nan
        else specs["pf_ratio_patient_id_col"].split()
    )
    pf_encounter_id_col = (
        []
        if specs["pf_ratio_encounter_id_col"] is np.nan
        else specs["pf_ratio_encounter_id_col"].split()
    )
    pf_icu_id_col = (
        []
        if specs["pf_ratio_icu_id_col"] is np.nan
        else specs["pf_ratio_icu_id_col"].split()
    )
    pf_hosp_icu_time_cols = (
        []
        if specs["pf_ratio_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["pf_ratio_hosp_and_or_icu_timestamp_cols"].split()
    )
    pf_vent_time_cols = (
        []
        if specs["pf_ratio_vent_timestamp_col"] is np.nan
        else specs["pf_ratio_vent_timestamp_col"].split()
    )
    pf_value_time_cols = specs["pf_ratio_value_timestamp_cols"].split()
    pf_value_cols = specs["pf_ratio_value_col"].split()
    pf_time_cols = pf_hosp_icu_time_cols + pf_vent_time_cols + pf_value_time_cols
    pf_id_col = []

    # Converting time columns
    try:
        pf_ratios[pf_time_cols] = pf_ratios[pf_time_cols].apply(pd.to_datetime)
        print("Columns with dates successfully converted to pd.datetime.")
    except TypeError:
        print(
            "Failed to convert columns that contain dates into datetime. Manually check."
        )

    # Renaming columns: IDs
    if len(pf_patient_id_col) == 1:
        pf_ratios = pf_ratios.rename(columns={pf_patient_id_col[0]: "patient_id"})
        pf_id_col.append("patient_id")
        print("Patient ID column renamed successfully!")
    else:
        print("No patient ID column specified in PF ratio table.")

    if len(pf_encounter_id_col) == 1:
        pf_ratios = pf_ratios.rename(columns={pf_encounter_id_col[0]: "encounter_id"})
        pf_id_col.append("encounter_id")
        print("Hospitalization/Encounter ID column renamed successfully!")
    else:
        print("No hospitalization/encounter ID column specified in PF ratio table.")

    if len(pf_icu_id_col) == 1:
        pf_ratios = pf_ratios.rename(columns={pf_icu_id_col[0]: "icu_id"})
        pf_id_col.append("icu_id")
        print("ICU stay ID column renamed successfully!")
    else:
        print("No ICU stay ID column specified in PF ratio table.")

    # Renaming columns: PF ratio timestamps
    if len(pf_value_time_cols) > 1:
        print(
            "More than one column contains the PF ratio timestamp. This requires case-by-case handling."
        )

    elif len(pf_value_time_cols) == 1:
        pf_ratios = pf_ratios.rename(
            columns={pf_value_time_cols[0]: "pf_ratio_timestamp"}
        )
        print("PF ratio timestamp column renamed successfully!")

    else:
        raise ValueError(
            "No timestamp for the PF ratio detected. Berlin definition cannot be applied."
        )

    # Renaming columns: PF ratio values
    if len(pf_value_cols) > 1:
        print(
            """
            More than one column contains the value.
            Likely, the value is one among many specified in the table.
            This requires case-by-case handling.
            """
        )

    elif len(pf_value_cols) == 1:
        pf_ratios = pf_ratios.rename(columns={pf_value_cols[0]: "pf_ratio_value"})
        print("PF ratio value column renamed successfully!")

    else:
        raise ValueError(
            "No column for PF ratio detected. Berlin definition cannot be applied."
        )

    # Renaming columns: Intubation timestamps
    if len(pf_vent_time_cols) > 1:
        print(
            "More than one column contains the ventilation start timestamp. This requires case-by-case handling."
        )

    elif len(pf_vent_time_cols) == 1:
        pf_ratios = pf_ratios.rename(
            columns={pf_vent_time_cols[0]: "vent_start_timestamp"}
        )
        print("Ventilation start timestamp column renamed successfully!")

    else:
        raise ValueError(
            "No timestamp for start of ventilation detected. Berlin definition cannot be applied."
        )

    # Make column names lowercase
    pf_ratios.columns = pf_ratios.columns.str.lower()

    # Dropping duplicates and nulls (from appropriate columns)
    pf_ratios = pf_ratios.drop_duplicates()

    if len(pf_value_time_cols) == 1 and len(pf_value_cols) == 1:
        pf_ratios = pf_ratios.dropna(
            subset=["pf_ratio_timestamp", "pf_ratio_value"], how="all"
        )
        print(
            "Successfully removed rows where both PF ratio timestamp and value are NULL."
        )
    else:
        print(
            "There's multiple PF ratio timestamp or value columns. Manually delete NULLs if desired."
        )

    # Merging to dictionary table if present
    if dictionary is not None:
        try:
            pf_ratios = pd.merge(dictionary, pf_ratios, how="right")
            print("Right merge with dictionary table was successful!")
        except MergeError:
            print(
                "Right merge with dictionary table did not work. Likely, there were no common columns to merge on."
            )
    else:
        pass

    # Sorting by IDs and then value timestamps
    pf_value_time_cols = [
        value_time_col.lower() for value_time_col in pf_value_time_cols
    ]
    if len(pf_value_time_cols) == 1:
        pf_ratios = pf_ratios.sort_values(by=pf_id_col + ["pf_ratio_timestamp"])
        print("Successfully sorted table by IDs and then by PF ratio timestamp.")
    else:
        pf_ratios = pf_ratios.sort_values(by=pf_id_col + pf_value_time_cols)
        print("Successfully sorted table by IDs and then by the PF ratio timestamps.")

    ### PEEP ###
    if peep is not None:
        print("\n\nDealing with PEEP table")
        peep_patient_id_col = (
            []
            if specs["peep_patient_id_col"] is np.nan
            else specs["peep_patient_id_col"].split()
        )
        peep_encounter_id_col = (
            []
            if specs["peep_encounter_id_col"] is np.nan
            else specs["peep_encounter_id_col"].split()
        )
        peep_icu_id_col = (
            []
            if specs["peep_icu_id_col"] is np.nan
            else specs["peep_icu_id_col"].split()
        )
        peep_hosp_icu_time_cols = (
            []
            if specs["peep_hosp_and_or_icu_timestamp_cols"] is np.nan
            else specs["peep_hosp_and_or_icu_timestamp_cols"].split()
        )
        peep_value_time_cols = specs["peep_value_timestamp_cols"].split()
        peep_value_cols = specs["peep_value_col"].split()
        peep_time_cols = peep_hosp_icu_time_cols + peep_value_time_cols
        peep_id_col = []

        # Converting time columns
        try:
            peep[peep_time_cols] = peep[peep_time_cols].apply(pd.to_datetime)
            print("Columns with dates successfully converted to pd.datetime.")
        except TypeError:
            print(
                "Failed to convert columns that contain dates into datetime. Manually check."
            )

        # Renaming columns: IDs
        if len(peep_patient_id_col) == 1:
            peep = peep.rename(columns={peep_patient_id_col[0]: "patient_id"})
            peep_id_col.append("patient_id")
            print("Patient ID column renamed successfully!")
        else:
            print("No patient ID column specified in PEEP table.")

        if len(peep_encounter_id_col) == 1:
            peep = peep.rename(columns={peep_encounter_id_col[0]: "encounter_id"})
            peep_id_col.append("encounter_id")
            print("Hospitalization/Encounter ID column renamed successfully!")
        else:
            print("No hospitalization/encounter ID column specified in PEEP table.")

        if len(peep_icu_id_col) == 1:
            peep = peep.rename(columns={peep_icu_id_col[0]: "icu_id"})
            peep_id_col.append("icu_id")
            print("ICU stay ID column renamed successfully!")
        else:
            print("No ICU stay ID column specified in PEEP table.")

        # Renaming columns: PEEP timestamps
        if len(peep_value_time_cols) > 1:
            print(
                "More than one column contains the PEEP timestamp. This requires case-by-case handling."
            )

        elif len(peep_value_time_cols) == 1:
            peep = peep.rename(columns={peep_value_time_cols[0]: "peep_timestamp"})
            print("PEEP timestamp column renamed successfully!")

        else:
            raise ValueError(
                "No timestamp for PEEP detected. Berlin definition cannot be applied."
            )

        # Renaming columns: PEEP values
        if len(peep_value_cols) > 1:
            print(
                """
                More than one column contains the value.
                Likely, the value is one among many specified in the table.
                This requires case-by-case handling.
                """
            )

        elif len(peep_value_cols) == 1:
            peep = peep.rename(columns={peep_value_cols[0]: "peep_value"})
            print("PEEP value column renamed successfully!")

        else:
            raise ValueError(
                "No column for PEEP detected. Berlin definition cannot be applied."
            )

        # Make column names lowercase
        peep.columns = peep.columns.str.lower()

        # Dropping duplicates and nulls (from appropriate columns)
        peep = peep.drop_duplicates()

        if len(peep_value_time_cols) == 1 and len(peep_value_cols) == 1:
            peep = peep.dropna(subset=["peep_timestamp", "peep_value"], how="all")
            print(
                "Successfully removed rows where both PEEP timestamp and value are NULL."
            )
        else:
            print(
                "There's multiple PEEP timestamp or value columns. Manually delete NULLs if desired."
            )

        # Merging to dictionary table if present
        if dictionary is not None:
            try:
                peep = pd.merge(dictionary, peep, how="right")
                print("Right merge with dictionary table was successful!")
            except MergeError:
                print(
                    "Right merge with dictionary table did not work. Likely, there were no common columns to merge on."
                )
        else:
            pass

        # Sorting by IDs and then value timestamps
        peep_value_time_cols = [
            value_time_col.lower() for value_time_col in peep_value_time_cols
        ]
        if len(peep_value_time_cols) == 1:
            peep = peep.sort_values(by=peep_id_col + ["peep_timestamp"])
            print("Successfully sorted table by IDs and then by PEEP timestamp.")
        else:
            peep = peep.sort_values(by=peep_id_col + peep_value_time_cols)
            print("Successfully sorted table by IDs and then by the PEEP timestamps.")

    else:
        pass

    ### Chest X-ray reports ###
    print("\n\nDealing with chest X-ray report table")
    cxr_patient_id_col = (
        []
        if specs["cxr_patient_id_col"] is np.nan
        else specs["cxr_patient_id_col"].split()
    )
    cxr_encounter_id_col = (
        []
        if specs["cxr_encounter_id_col"] is np.nan
        else specs["cxr_encounter_id_col"].split()
    )
    cxr_icu_id_col = (
        [] if specs["cxr_icu_id_col"] is np.nan else specs["cxr_icu_id_col"].split()
    )
    cxr_hosp_icu_time_cols = (
        []
        if specs["cxr_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["cxr_hosp_and_or_icu_timestamp_cols"].split()
    )
    cxr_value_time_cols = specs["cxr_value_timestamp_cols"].split()
    cxr_value_cols = specs["cxr_value_col"].split()
    cxr_time_cols = cxr_hosp_icu_time_cols + cxr_value_time_cols
    cxr_label_cols = (
        []
        if specs["cxr_value_label_cols"] is np.nan
        else specs["cxr_value_label_cols"].split()
    )
    cxr_id_col = []

    # Converting time columns
    try:
        cxr_reports[cxr_time_cols] = cxr_reports[cxr_time_cols].apply(pd.to_datetime)
        print("Columns with dates successfully converted to pd.datetime.")
    except TypeError:
        print(
            "Failed to convert columns that contain dates into datetime. Manually check."
        )

    # Renaming columns: IDs
    if len(cxr_patient_id_col) == 1:
        cxr_reports = cxr_reports.rename(columns={cxr_patient_id_col[0]: "patient_id"})
        cxr_id_col.append("patient_id")
        print("Patient ID column renamed successfully!")
    else:
        print("No patient ID column specified in chest X-ray report table.")

    if len(cxr_encounter_id_col) == 1:
        cxr_reports = cxr_reports.rename(
            columns={cxr_encounter_id_col[0]: "encounter_id"}
        )
        cxr_id_col.append("encounter_id")
        print("Hospitalization/Encounter ID column renamed successfully!")
    else:
        print(
            "No hospitalization/encounter ID column specified in chest X-ray report table."
        )

    if len(cxr_icu_id_col) == 1:
        cxr_reports = cxr_reports.rename(columns={cxr_icu_id_col[0]: "icu_id"})
        cxr_id_col.append("icu_id")
        print("ICU stay ID column renamed successfully!")
    else:
        print("No ICU stay ID column specified in chest X-ray report table.")

    # Renaming columns: CXR report timestamp
    if len(cxr_value_time_cols) > 1:
        print(
            """
            More than one column contains the chest X-ray report timestamp.
            This requires case-by-case handling.
            """
        )

    elif len(cxr_value_time_cols) == 1:
        cxr_reports = cxr_reports.rename(
            columns={cxr_value_time_cols[0]: "cxr_timestamp"}
        )
        print("Chest X-ray report timestamp column renamed successfully!")

    else:
        raise ValueError(
            "No timestamp for the Chest X-ray report detected. Berlin definition cannot be applied."
        )

    # Renaming columns: CXR report text
    if len(cxr_value_cols) > 1:
        print(
            """
            More than one column contains the value.
            Likely, the value is one among many specified in the table.
            This requires case-by-case handling.
            """
        )

    elif len(cxr_value_cols) == 1:
        cxr_reports = cxr_reports.rename(columns={cxr_value_cols[0]: "cxr_text"})
        print("Chest X-ray report text column renamed successfully!")

    else:
        raise ValueError(
            "No column for chest X-ray text detected. Berlin definition cannot be applied."
        )

    # Renaming columns: CXR report labels or scores
    if len(cxr_label_cols) > 1:
        print(
            """
            More than one column contains labels/scores/ratings.
            Likely, the are multiple human raters, and then one tie-breaker.
            This requires case-by-case handling.
            """
        )

    elif len(cxr_label_cols) == 1:
        cxr_reports = cxr_reports.rename(columns={cxr_label_cols[0]: "cxr_score"})
        print("Chest X-ray report label/score column renamed successfully!")

    else:
        print("This CXR table isn't annotated/labelled/scored (however you call it).")

    # Make column names lowercase
    cxr_reports.columns = cxr_reports.columns.str.lower()

    # Dropping duplicates and nulls (from appropriate columns)
    cxr_reports = cxr_reports.drop_duplicates()

    if len(cxr_value_time_cols) == 1 and len(cxr_value_cols) == 1:
        cxr_reports = cxr_reports.dropna(
            subset=["cxr_timestamp", "cxr_text"], how="all"
        )
        print(
            "Successfully removed rows where both chest X-ray timestamp and text are NULL."
        )
    else:
        print(
            "There's multiple chest X-ray timestamp or text columns. Manually delete NULLs if desired."
        )

    # Merging to dictionary table if present
    if dictionary is not None:
        try:
            cxr_reports = pd.merge(dictionary, cxr_reports, how="right")
            print("Right merge with dictionary table was successful!")
        except MergeError:
            print(
                "Right merge with dictionary table did not work. Likely, there were no common columns to merge on."
            )
    else:
        pass

    # Sorting by IDs and then value timestamps
    cxr_value_time_cols = [
        value_time_col.lower() for value_time_col in cxr_value_time_cols
    ]
    if len(cxr_value_time_cols) == 1:
        cxr_reports = cxr_reports.sort_values(by=cxr_id_col + ["cxr_timestamp"])
        print("Successfully sorted table by IDs and then by CXR timestamp.")
    else:
        cxr_reports = cxr_reports.sort_values(by=cxr_id_col + cxr_value_time_cols)
        print("Successfully sorted table by IDs and then by the CXR timestamps.")

    ### Attending physician notes ###
    print("\n\nDealing with attending physician notes table")
    notes_patient_id_col = (
        []
        if specs["notes_patient_id_col"] is np.nan
        else specs["notes_patient_id_col"].split()
    )
    notes_encounter_id_col = (
        []
        if specs["notes_encounter_id_col"] is np.nan
        else specs["notes_encounter_id_col"].split()
    )
    notes_icu_id_col = (
        [] if specs["notes_icu_id_col"] is np.nan else specs["notes_icu_id_col"].split()
    )
    notes_hosp_icu_time_cols = (
        []
        if specs["notes_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["notes_hosp_and_or_icu_timestamp_cols"].split()
    )
    notes_value_time_cols = specs["notes_value_timestamp_cols"].split()
    notes_value_cols = specs["notes_value_col"].split()
    notes_time_cols = notes_hosp_icu_time_cols + notes_value_time_cols
    notes_id_col = []

    # Converting time columns
    try:
        attn_notes[notes_time_cols] = attn_notes[notes_time_cols].apply(pd.to_datetime)
        print("Columns with dates successfully converted to pd.datetime.")
    except TypeError:
        print(
            "Failed to convert columns that contain dates into datetime. Manually check."
        )

    # Renaming columns: IDs
    if len(notes_patient_id_col) == 1:
        attn_notes = attn_notes.rename(columns={notes_patient_id_col[0]: "patient_id"})
        notes_id_col.append("patient_id")
        print("Patient ID column renamed successfully!")
    else:
        print("No patient ID column specified in attending notes table.")

    if len(notes_encounter_id_col) == 1:
        attn_notes = attn_notes.rename(
            columns={notes_encounter_id_col[0]: "encounter_id"}
        )
        notes_id_col.append("encounter_id")
        print("Hospitalization/Encounter ID column renamed successfully!")
    else:
        print(
            "No hospitalization/encounter ID column specified in attending notes table."
        )

    if len(notes_icu_id_col) == 1:
        attn_notes = attn_notes.rename(columns={notes_icu_id_col[0]: "icu_id"})
        notes_id_col.append("icu_id")
        print("ICU stay ID column renamed successfully!")
    else:
        print("No ICU stay ID column specified in attending notes table.")

    # Renaming columns: Note timestamp
    if len(notes_value_time_cols) > 1:
        print(
            "More than one column contains the attending notes timestamp. This requires case-by-case handling."
        )

    elif len(notes_value_time_cols) == 1:
        attn_notes = attn_notes.rename(
            columns={notes_value_time_cols[0]: "notes_timestamp"}
        )
        print("Attending physician notes timestamp column renamed successfully!")

    else:
        raise ValueError(
            "No timestamp for the attending physician notes detected. Berlin definition cannot be applied."
        )

    # Renaming columns: Note text
    if len(notes_value_cols) > 1:
        print(
            """
            More than one column contains the value.
            Likely, the value is one among many specified in the table.
            This requires case-by-case handling.
            """
        )

    elif len(notes_value_cols) == 1:
        attn_notes = attn_notes.rename(columns={notes_value_cols[0]: "notes_text"})
        print("Attending physician notes text column renamed successfully!")

    else:
        raise ValueError(
            "No column for chest X-ray text detected. Berlin definition cannot be applied."
        )

    # Make column names lowercase
    attn_notes.columns = attn_notes.columns.str.lower()

    # Dropping duplicates and nulls (from appropriate columns)
    attn_notes = attn_notes.drop_duplicates()

    if len(notes_value_time_cols) == 1 and len(notes_value_cols) == 1:
        attn_notes = attn_notes.dropna(
            subset=["notes_timestamp", "notes_text"], how="all"
        )
        print(
            "Successfully removed rows where both attending note timestamp and text are NULL."
        )
    else:
        print(
            "There's multiple attending note timestamp or text columns. Manually delete NULLs if desired."
        )

    # Merging to dictionary table if present
    if dictionary is not None:
        try:
            attn_notes = pd.merge(dictionary, attn_notes, how="right")
            print("Right merge with dictionary table was successful!")
        except MergeError:
            print(
                "Right merge with dictionary table did not work. Likely, there were no common columns to merge on."
            )
    else:
        pass

    # Sorting by IDs and then value timestamps
    notes_value_time_cols = [
        value_time_col.lower() for value_time_col in notes_value_time_cols
    ]
    if len(notes_value_time_cols) == 1:
        attn_notes = attn_notes.sort_values(by=notes_id_col + ["notes_timestamp"])
        print("Successfully sorted table by IDs and then by attending note timestamp.")
    else:
        attn_notes = attn_notes.sort_values(by=notes_id_col + notes_value_time_cols)
        print(
            "Successfully sorted table by IDs and then by the attending note timestamps."
        )

    ### Echocardiography reports ###
    print("\n\nDealing with ECHO reports table")
    echo_patient_id_col = (
        []
        if specs["echo_patient_id_col"] is np.nan
        else specs["echo_patient_id_col"].split()
    )
    echo_encounter_id_col = (
        []
        if specs["echo_encounter_id_col"] is np.nan
        else specs["echo_encounter_id_col"].split()
    )
    echo_icu_id_col = (
        [] if specs["echo_icu_id_col"] is np.nan else specs["echo_icu_id_col"].split()
    )
    echo_hosp_icu_time_cols = (
        []
        if specs["echo_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["echo_hosp_and_or_icu_timestamp_cols"].split()
    )
    echo_value_time_cols = specs["echo_value_timestamp_cols"].split()
    echo_value_cols = specs["echo_value_col"].split()
    echo_time_cols = echo_hosp_icu_time_cols + echo_value_time_cols
    echo_id_col = []

    # Converting time columns
    try:
        echo_reports[echo_time_cols] = echo_reports[echo_time_cols].apply(
            pd.to_datetime
        )
        print("Columns with dates successfully converted to pd.datetime.")
    except TypeError:
        print(
            "Failed to convert columns that contain dates into datetime. Manually check."
        )

    # Renaming columns: IDs
    if len(echo_patient_id_col) == 1:
        echo_reports = echo_reports.rename(
            columns={echo_patient_id_col[0]: "patient_id"}
        )
        echo_id_col.append("patient_id")
        print("Patient ID column renamed successfully!")
    else:
        print("No patient ID column specified in ECHO report table.")

    if len(echo_encounter_id_col) == 1:
        echo_reports = echo_reports.rename(
            columns={echo_encounter_id_col[0]: "encounter_id"}
        )
        echo_id_col.append("encounter_id")
        print("Hospitalization/Encounter ID column renamed successfully!")
    else:
        print("No hospitalization/encounter ID column specified in ECHO report table.")

    if len(echo_icu_id_col) == 1:
        echo_reports = echo_reports.rename(columns={echo_icu_id_col[0]: "icu_id"})
        echo_id_col.append("icu_id")
        print("ICU stay ID column renamed successfully!")
    else:
        print("No ICU stay ID column specified in ECHO report table.")

    # Renaming columns: ECHO report timestamp
    if len(echo_value_time_cols) > 1:
        print(
            "More than one column contains the ECHO report timestamp. This requires case-by-case handling."
        )

    elif len(echo_value_time_cols) == 1:
        echo_reports = echo_reports.rename(
            columns={echo_value_time_cols[0]: "echo_timestamp"}
        )
        print("ECHO report timestamp column renamed successfully!")

    else:
        raise ValueError(
            "No timestamp for the ECHO report  detected. Berlin definition cannot be applied."
        )

    # Renaming columns: ECHO report text
    if len(echo_value_cols) > 1:
        print(
            """
            More than one column contains the value.
            Likely, the value is one among many specified in the table.
            This requires case-by-case handling.
            """
        )

    elif len(echo_value_cols) == 1:
        echo_reports = echo_reports.rename(columns={echo_value_cols[0]: "echo_text"})
        print("ECHO report text column renamed successfully!")

    else:
        raise ValueError(
            "No column for the ECHO report text detected. Berlin definition cannot be applied."
        )

    # Make column names lowercase
    echo_reports.columns = echo_reports.columns.str.lower()

    # Dropping duplicates and nulls (from appropriate columns)
    echo_reports = echo_reports.drop_duplicates()

    if len(echo_value_time_cols) == 1 and len(echo_value_cols) == 1:
        echo_reports = echo_reports.dropna(
            subset=["echo_timestamp", "echo_text"], how="all"
        )
        print(
            "Successfully removed rows where both ECHO report timestamp and text are NULL."
        )
    else:
        print(
            """
            There's multiple ECHO report timestamp or text columns.
            Manually delete NULLs if desired.
            """
        )

    # Merging to dictionary table if present
    if dictionary is not None:
        try:
            echo_reports = pd.merge(dictionary, echo_reports, how="right")
            print("Right merge with dictionary table was successful!")
            
        except MergeError:
            print(
                """
                Right merge with dictionary table did not work.
                Likely, there were no common columns to merge on.
                """
            )
    else:
        pass

    # Sorting by IDs and then value timestamps
    echo_value_time_cols = [
        value_time_col.lower() for value_time_col in echo_value_time_cols
    ]
    if len(echo_value_time_cols) == 1:
        echo_reports = echo_reports.sort_values(by=echo_id_col + ["echo_timestamp"])
        print("Successfully sorted table by IDs and then by ECHO report timestamp.")
    else:
        echo_reports = echo_reports.sort_values(by=echo_id_col + echo_value_time_cols)
        print(
            "Successfully sorted table by IDs and then by the ECHO report timestamps."
        )

    ### Brain/beta natriuretic peptide ###
    print("\n\nDealing with BNP values table")
    bnp_patient_id_col = (
        []
        if specs["bnp_patient_id_col"] is np.nan
        else specs["bnp_patient_id_col"].split()
    )
    bnp_encounter_id_col = (
        []
        if specs["bnp_encounter_id_col"] is np.nan
        else specs["bnp_encounter_id_col"].split()
    )
    bnp_icu_id_col = (
        [] if specs["bnp_icu_id_col"] is np.nan else specs["bnp_icu_id_col"].split()
    )
    bnp_hosp_icu_time_cols = (
        []
        if specs["bnp_hosp_and_or_icu_timestamp_cols"] is np.nan
        else specs["bnp_hosp_and_or_icu_timestamp_cols"].split()
    )
    bnp_value_time_cols = specs["bnp_value_timestamp_cols"].split()
    bnp_value_cols = specs["bnp_value_col"].split()
    bnp_time_cols = bnp_hosp_icu_time_cols + bnp_value_time_cols
    bnp_id_col = []

    # Converting time columns
    try:
        bnp_values[bnp_time_cols] = bnp_values[bnp_time_cols].apply(pd.to_datetime)
        print("Columns with dates successfully converted to pd.datetime.")
    except TypeError:
        print(
            "Failed to convert columns that contain dates into datetime. Manually check."
        )

    # Renaming columns: IDs
    if len(bnp_patient_id_col) == 1:
        bnp_values = bnp_values.rename(columns={bnp_patient_id_col[0]: "patient_id"})
        bnp_id_col.append("patient_id")
        print("Patient ID column renamed successfully!")
    else:
        print("No patient ID column specified in BNP table.")

    if len(bnp_encounter_id_col) == 1:
        bnp_values = bnp_values.rename(
            columns={bnp_encounter_id_col[0]: "encounter_id"}
        )
        bnp_id_col.append("encounter_id")
        print("Hospitalization/Encounter ID column renamed successfully!")
    else:
        print("No hospitalization/encounter ID column specified in BNP table.")

    if len(bnp_icu_id_col) == 1:
        bnp_values = bnp_values.rename(columns={bnp_icu_id_col[0]: "icu_id"})
        bnp_id_col.append("icu_id")
        print("ICU stay ID column renamed successfully!")
    else:
        print("No ICU stay ID column specified in BNP table.")

    # Renaming columns: BNP timestamp
    if len(bnp_value_time_cols) > 1:
        print(
            "More than one column contains the BNP timestamp. This requires case-by-case handling."
        )

    elif len(bnp_value_time_cols) == 1:
        bnp_values = bnp_values.rename(
            columns={bnp_value_time_cols[0]: "bnp_timestamp"}
        )
        print("BNP timestamp column renamed successfully!")

    else:
        raise ValueError(
            "No timestamp for the BNP value detected. Berlin definition cannot be applied."
        )

    # Renaming columns: BNP value
    if len(bnp_value_cols) > 1:
        print(
            """
            More than one column contains the value.
            Likely, the value is one among many specified in the table.
            This requires case-by-case handling.
            """
        )

    elif len(bnp_value_cols) == 1:
        bnp_values = bnp_values.rename(columns={bnp_value_cols[0]: "bnp_value"})
        print("BNP value column renamed successfully!")

    else:
        raise ValueError(
            "No column for the BNP value detected. Berlin definition cannot be applied."
        )

    # Make column names lowercase
    bnp_values.columns = bnp_values.columns.str.lower()

    # Dropping duplicates and nulls (from appropriate columns)
    bnp_values = bnp_values.drop_duplicates()

    if len(bnp_value_time_cols) == 1 and len(bnp_value_cols) == 1:
        bnp_values = bnp_values.dropna(subset=["bnp_timestamp", "bnp_value"], how="all")
        print("Successfully removed rows where both BNP timestamp and value are NULL.")
    else:
        print(
            "There's multiple echo report timestamp or text columns. Manually delete NULLs if desired."
        )

    # Merging to dictionary table if present
    if dictionary is not None:
        try:
            bnp_values = pd.merge(dictionary, bnp_values, how="right")
            print("Right merge with dictionary table was successful!")
        except MergeError:
            print(
                "Right merge with dictionary table did not work. Likely, there were no common columns to merge on."
            )
    else:
        pass

    # Sorting by IDs and then value timestamps
    bnp_value_time_cols = [
        value_time_col.lower() for value_time_col in bnp_value_time_cols
    ]
    if len(bnp_value_time_cols) == 1:
        bnp_values = bnp_values.sort_values(by=bnp_id_col + ["bnp_timestamp"])
        print("Successfully sorted table by IDs and then by BNP timestamp.")
    else:
        bnp_values = bnp_values.sort_values(by=bnp_id_col + bnp_value_time_cols)
        print("Successfully sorted table by IDs and then by the BNP timestamps.")

    print("\n\nAll set! Check your tables\n\n")

    return (
        dictionary,
        pf_ratios,
        peep,
        cxr_reports,
        attn_notes,
        echo_reports,
        bnp_values,
    )


def detect_datetime_and_convert(df, columns_returned="all"):
    """
    Infers which columns in pandas dataframe contain date or datetime entries,
    converts them, and returns a pandas dataframe specified by user.

    df - pandas dataframe of interest
    columns_returned - Whether to return the entire converted dataframe, or a subset
        'all' - Default. Return the entire dataframe with converted columns
        'datetime' - Return just the converted columns
        'no_datetime' - Return dataframe without converted columns
    """

    # Captures various date formats using a dash (-) to separate
    mask1 = df.astype(str).apply(
        lambda x: x.str.fullmatch(
            r"(?<!\S)\d{1,4}-\d{1,2}-\d{2,4}(\s*\d{0,2}\:\d{0,2}\:\d{0,2}|\s*\d{0,2}\:\d{0,2}|)(?!\S)"
        ).any()
    )

    # Captures various date formats using a slash (/) to separate
    mask2 = df.astype(str).apply(
        lambda x: x.str.fullmatch(
            r"(?<!\notes1S)\d{1,4}\/\d{1,2}\/\d{2,4}(\s*\d{0,2}\:\d{0,2}\:\d{0,2}|\s*\d{0,2}\:\d{0,2}|)(?!\S)"
        ).any()
    )

    mask = mask1 | mask2

    df.loc[:, mask] = df.loc[:, mask].apply(pd.to_datetime, errors="coerce")

    if columns_returned == "all":
        return df

    if columns_returned == "datetime":
        return df.loc[:, mask]

    if columns_returned == "no_datetime":
        return df.loc[:, ~mask]

    raise ValueError(
        "columns_returned parameter invalid. Pass one of 'all', 'datetime', or 'no_datetime'"
        )
