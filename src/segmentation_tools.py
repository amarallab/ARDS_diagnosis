import re
from operator import itemgetter
from pathlib import Path


def process_times(note, pattern):
    """
    Look for presence of times, in 12h clock or 24h clock,
    using a regular expression pattern and replace by format aahbb

    inputs:
        note -- string with cxr report
        pattern -- re string

    outputs:
        note -- string with cxr report
    """

    while True:
        match = re.search(pattern, note)
        if not match:
            return note

        flag = False
        text = note[match.start() : match.end()]
        hours, minutes = text.split(":")
        hours = int(hours)
        if "p" in minutes:
            flag = True
        minutes = int(minutes.rstrip("pam. "))
        if hours < 12 and flag:
            hours += 12

        note = note.replace(text, f" {hours}h{minutes} ")

    return note


def clean_up_string(text):
    """
    Clean up some issues with text (missing spaces, commas, and so on)

    Input:
        text -- string with cxr report

    Output:
        note -- cleaned up text
    """

    note = text.lower()

    # Replace new lines by a period and all whitespaces by a single space.
    note = note.replace("\n", " . ")
    note = " ".join(note.split())

    # Replace different time standards with single standard (24h clock as xxhmm)
    pattern_12h = r"((1[0-2])|([0\s]\d)):[0-5]\d\s*[ap][\s.]*m[\s.]*"
    note = process_times(note, pattern_12h)
    pattern_24h = r"((2[0-3])|(1\d)|((0|\s)\d)):[0-5]\d\s*"
    note = process_times(note, pattern_24h)

    # Add a space after a colon, remove parenthesis an brackets, and
    # replace / by a single space
    note = note.replace(":", ": ")
    note = note.replace(";", ". ")
    note = note.replace(",", " ")
    note = note.replace('"', " ")
    note = note.replace("'", " ")
    note = note.replace("(", "").replace(")", "")
    note = note.replace("[", "").replace("]", "")
    note = note.replace("/", "_")

    # Fix words for which in cohort 2 reports there is merging of words
    problematic_words = [
        "comparison:",
        "findings:",
        "technique:",
        "impression:",
        "conclusions:",
        "history:",
        "key:",
        "findings_conclusion:",
        "procedure",
        "comment",
    ]

    for word in problematic_words:
        note = note.replace(word, " " + word)

    # Replace plurals by singular version of word for section titles
    section_words = {
        "comparisons": "comparison",
        "conclusions": "conclusion",
        "comments": "comment",
        "findings": "finding",
        "impressions": "impression",
        "indications": "indication",
        "limitations": "limitation",
        "procedures": "procedure",
    }

    for key, value in section_words.items():
        note = note.replace(key, value)

    return note


def remove_easy_sections(report, section_order, verbose=False):
    """
    Find section names in report by looking for terms followed by colon and remove target_sections
    from text.

    Input:
        report -- string with cxr report
        section_order -- list of tuples with section name and Boolean about keeping it or not
        verbose -- boolean to activate printing, default is False

    Output:
        note -- string with processed cxr report
    """

    note = clean_up_string(report)

    if verbose:
        print(f"\n{note}\n")

    # Create list with indices of start of all sections
    reasons_for_concern = True
    section_ranges = []
    for section in section_order:
        match = re.search(section[0], note)

        if match is None:
            section_ranges.append(None)
        else:
            if section[0] in [
                "finding:",
                "finding_conclusion:",
                "conclusion:",
                "impression",
            ]:
                reasons_for_concern = False
            section_ranges.append(match.start())

    section_ranges.append(len(note) - 1)

    if verbose:
        if len(section_order) != len(section_ranges) - 1:
            print("Issue with section ranges!")
        names, states = zip(*section_order)
        print(names)
        print(states)
        print(section_ranges)

    if reasons_for_concern:
        if verbose:
            print("--------\nThere were reasons for concern.\n")
        return note

    # Remove text of target sections by using indices delimiting said sections
    strings_to_be_removed = []
    for i, section in enumerate(section_order):
        if not section[1] and section_ranges[i]:
            index_i = section_ranges[i]
            step = 1
            while section_ranges[i + step] is None:
                step += 1
            index_f = section_ranges[i + step]
            strings_to_be_removed.append(note[index_i:index_f])

    for i, pattern in enumerate(strings_to_be_removed):
        if verbose:
            print(f"\n\n{i:2}---------{pattern}")
        note = note.replace(pattern, "")

    return note


def handle_subsection_titles(report):
    """
    Find lines with subsection titles, split them, and connect them

    Input:
        report -- string with cxr report

    Output:
        new_report -- list of strings
    """

    new_report = []

    for line in report:
        if ":" in line:
            parts = line.split(":")
            for k in range(len(parts) - 1):
                new_report.append((parts[k] + ":").strip())
            new_report.append(parts[-1].strip())
        else:
            new_report.append(line.strip())

    return new_report


def remove_lines_on_other_organs(report, exclusion_set):
    """
    Find statements with mentions of terms indicative of organs other than lungs

    Input:
        report -- list of statements (strings)
        excluded_set -- list of strings

    Output:
        new_report -- list of strings
    """

    new_report = []

    for line in report:
        if ":" in line:
            new_report.append(line)

        else:
            if not exclusion_set.intersection(set(line.split())):
                new_report.append(line)

    return new_report


def curate_indicator_word_list(filename, targeted_stemming, verbose=False):
    """
    Reads in a file containing raw indicator words of bilateral
    infiltrates in chest X-rays, and processes them to return
    a clean list if such words.

    Input:
        - filename: str, name of the file containing raw words
        - targeted_stemming: dict, specific key:value pairs for stemming

    Returns:
        - indicator_words: list of str, cleaned/stemmed indicator words
    """

    filepath = Path("./") / "src" / f"{filename}.txt"

    indicators = []
    with open(filepath, "r") as file1:
        for line in file1.readlines():
            indicators.append(line.strip().split(","))

    if verbose:
        print(f"Loaded {len(indicators)} indicator terms.\n")

    # Stem words in indicator tuples
    clean_indicators = []
    for item in indicators:

        for j, word in enumerate(item):
            if word in targeted_stemming.keys():
                item[j] = targeted_stemming[word]

        if item not in clean_indicators:
            clean_indicators.append(item)

    clean_indicators.sort(key=itemgetter(0))

    if verbose:
        for item in clean_indicators:
            print(item)

    indicator_words = []
    for item in clean_indicators:
        for word in item:
            if word not in indicator_words:
                indicator_words.append(word)

    indicator_words.sort()

    if verbose:
        print(len(indicator_words))
        print(indicator_words)
        print("\nAdded extra words\n")

    indicator_words.extend(
        ["angle", "effusion", "patch", "space", "trace", "left", "right"]
    )
    indicator_words.sort()

    if verbose:
        print("\n", len(indicator_words), "\n")
        print(indicator_words)

    return indicator_words


def stem_indicator_words(report, targeted_stemming):
    """
    Find indicator words in statements and stem them

    Input:
        report -- list of strings
        indicator_words -- list of strings
        targeted_stemming -- dictionary with stemming replacements

    Output:
        new_report -- list of strings
    """

    new_report = []

    for line in report:
        stemmed_words = [
            targeted_stemming[w] if w in targeted_stemming.keys() else w
            for w in line.split()
        ]

        if len(stemmed_words) >= 1:
            line = " ".join(stemmed_words)
            new_report.append(line)

    return new_report


def remove_stopwords(report, complex_stopwords, simple_stopwords):
    """
    Find both complex and simple stopwords in statements and remove them

    Input:
        report -- list of strings
        complex_stopwords -- list of strings
        simple_stopwords -- list of string

    Output:
        new_report -- list of string
    """

    new_report = []

    for line in report:
        if ":" in line:
            new_report.append(line)

        else:
            for snippet in complex_stopwords:
                if snippet in line:
                    line = line.replace(snippet, " ")

            words = line.split()
            new_words = [w for w in words if w not in simple_stopwords]
            new_report.append(" ".join(new_words))

    return new_report


def remove_sections_n_duplicate_lines(report):
    """
    Find duplicate statements and section titles and remove them

    Input:
        report -- list of strings

    Output:
        new_report -- list of string
    """

    new_report = []

    for line in report:
        if line not in new_report and ":" not in line:
            new_report.append(line)

    return new_report


def refine_cleaning(report, useless_statements):
    """
    Find useless and numeric statements; remove them

    Input:
        report -- list of strings
        useless_statements -- list of strings

    Output:
        new_report -- list of string
    """

    new_report = []

    for line in report:
        logic = False
        if not line.isnumeric():

            # Check if at least one useless_statements is equal to line.
            # If so, set line for removal
            for item in useless_statements:
                if line.strip() == item.strip():
                    logic = True
                    break

        else:
            logic = True

        if not logic:
            new_report.append(line)

    return new_report


def remove_dictation(report, dictation_string, verbose=False):
    """
    Find dictation line and remove

    Input:
        report -- list of strings

    Output:
        new_report -- list of string
    """
    new_report = []

    for line in report:
        if line.startswith(dictation_string):
            if verbose:
                print("starts with")
        else:
            if verbose:
                print("doesn")
            new_report.append(line)

    return new_report


def extract_surroundings_of_risk_factor_and_process(
    risk_factor_label, text_field=None, add_column_name=None, verbose=False
):
    """
    Matches key words from risk factors,
    and adds a column with words surrounding the risk factor mention.

    Inputs:
    - risk_factor_label: list of dict, notes data with flags
                                        for a given risk factor.
    - text_field: str, specify name of column/field having the note.
    - add_column_name: str, specify risk factor name for naming new column.
                            name will follow pattern: "seg_{add_column_name}"
    """

    # patterns to search for
    patterns = {
        "pneumonia_pattern": r"(?<!\w)(?:PCPpneumonia|pneumonia|Pneumonia|PNEUMONIA|pneumoniae|pneunonia|pneunoniae|pnuemonia|bronchopneumonia|parapneumonic|PNA|CAP|VAP|HAP|HCAP|hcap|infection|abx|PCP)(?!\w)",
        "aspiration_pattern": r"(?i)(?<!\w)aspiration(?!\w)",
        "inhalation_pattern": r"(?i)(?<!\w)(?:inhaled|inhalation)(?!\w)",
        "pulm_contusion_pattern": r"(?i)(?<!\w)(?:pulmonary |pulmoanry )(?:contusion|contusions)(?!\w)",
        "vasculitis_pattern": r"(?i)(?<!\w)(?:pulmonary vasculitis|vasculitis)(?!\w)",
        "drowning_pattern": r"(?i)(?<!\w)(?:drowned|drowning)(?!\w)",
        "sepsis_pattern": r"(?i)(?<!\w)(?:sepsis|urosepsis|septiuc|septic|ssepsis|sseptic|spetic)(?!\w)",
        "shock_pattern": r"(?i)(?<!\w)(?:shock|shocks|schock)(?!\w)",
        "overdose_pattern": r"(?i)(?<!\w)(?:overdose|drug overdose)(?!\w)",
        "trauma_pattern": r"(?i)(?<!\w)(?<!OGT\s)(?:trauma|traumatic|barotrauma|barotraumatic)(?!\w)",
        "pancreatitis_pattern": r"(?i)(?<!\w)pancreatitis(?!\w)",
        "burn_pattern": r"(?i)(?<!\w)(?:burn|burns)(?!\w)",
        "trali_pattern": r"(?<!\w)(?:TRALI|(?i)transfusion(?:-|\s)related acute lung injury|(?i)transfusion(?:-|\s)associated acute lung injury)(?!\w)",
        "ards_pattern": r"(?i)(?<!\w)(?:ards|acute respiratory distress syndrome|acute lung injury|ali|ardsnet|acute hypoxemic respiratory failure)(?!\w)",
        "pregnant_pattern": r"(?i)(?<!\w)(?:IUP|G\dP\d)(?!\w)",
        "chf_pattern": r"(?i)(?<!\w)(?:congestive heart failure|chf|diastolic HF|systolic HF|heart failure|diastolic dysfunction|LV dysfunction|low cardiac output syndrome|low cardiac ouput syndrome|low CO state)(?!\w)",
        "cardiogenic_pattern": r"(?i)(?<!\w)(?<!non\s)(?<!non-)(?:cardiogenic|cardigenic|cardiogemic|cardiac pulmonary edema|cardiac and septic shock|Shock.{1,15}suspect.{1,15}RV failure)(?!\w)",
        "non_cardiogenic_pattern": r"(?i)(?<!\w)(?:non(?:-|\s)cardiogenic|noncardiogenic|non(?:-|\s)cardigenic|noncardigenic)(?!\w)",
        "palliative_pattern": r"(?i)(?<!\w)(?:palliative care|comfort care|withdraw care|comfort alone|withdraw support in favor of palliation)(?!\w)",
        "cardiac_arrest_pattern": r"(?i)(?<!\w)(?:arrest|cardiorespiratory arrest)(?!\w)",
        "dementia_pattern": r"(?i)(?<!\w)dementia(?!\w)",
        "stroke_pattern": r"(?i)(?<!\w)(?:stroke|strokes|cerebellar hemorrhage|intracerebral hemorrhage|BG hemorrhage|cva|cerebrovascular accident|cefrebellar infarcts\/basilar stenosis)(?!\w)",
        "alcohol_pattern": r"(?i)(?<!\w)(?:alcohol withdrawal|dts|dt''s|dt|alcohol dependence|alcohol abuse|etoh abuse|etoh withdrawal|etoh withdrawl|etoh w\/drawal|delirium tremens)(?!\w)",
        "seizure_pattern": r"(?i)(?<!\w)(?<!no e/o subclinical\s)(?<!no e/o subclinical )(?:seizure|seizures)(?!\w)",
        "ami_pattern": r"(?i)(?<!\w)(?:ami|acute myocardial ischemia|acute myocardial infarction|myocardial infarction|nstemi|non-st elevation mi|stemi|st elevation mi|acute mi)(?!\w)",
    }

    # Modify this variable to control how big of a section to extract surrounding a risk factor.
    window_size = 100

    spelling_correction = {
        "pneunonia": "pneumonia",
        "pneunoniae": "pneumoniae",
        "pnuemonia": "pneumonia",
        "septiuc": "septic",
        "ssepsis": "sepsis",
        "sseptic": "septic",
        "spetic": "septic",
        "schock": "shock",
        "cardigenic": "cardiogenic",
        "cardiogemic": "cardiogenic",
        "non-cardigenic": "non-cardiogenic",
        "non cardigenic": "non cardiogenic",
        "noncardigenic": "noncardiogenic",
        "dts": "delirium tremens",
        "dt''s": "delirium tremens",
        "dt": "delirium tremens",
        "etoh withdrawl": "etoh withdrawal",
        "etoh w\/drawal": "etoh withdrawal",
    }

    for key, value in patterns.items():
        if add_column_name in key:
            pattern = value
            break

    for i, record in enumerate(risk_factor_label):
        try:
            my_string = record[text_field]
        except KeyError:
            print(
                """
                Make sure you are using the right string for the attending note column,
                and try again.
                """
            )
            break

        if add_column_name in ("sepsis", "shock", "cardiac_arrest"):
            token_collect = []
        else:
            token_collect = set()

        for match in re.finditer(pattern, my_string):
            start = max([0, match.start() - window_size])
            end = min([len(my_string), match.end() + window_size])

            text = my_string[start:end].split(" ")
            text = [
                token.strip(":;,.-0123456789/()") for token in text
            ]  # Take away punctuation marks
            text = [
                token.strip().lower() for token in text
            ]  # Take away leading or trailing spaces

            for token in text:
                if token in spelling_correction:
                    if add_column_name in ("sepsis", "shock", "cardiac_arrest"):
                        # Correct identified typos
                        token_collect.append(
                            spelling_correction[token]
                            )
                    else:
                        # Correct identified typos
                        token_collect.add(
                            spelling_correction[token]
                            )
                else:
                    if add_column_name in ("sepsis", "shock", "cardiac_arrest"):
                        token_collect.append(token)
                    else:
                        token_collect.add(token)

        collected_text = " ".join(token_collect).strip()
        
        if len(collected_text) > 0:
            record[f"seg_{add_column_name}"] = collected_text
        else:
            if verbose:
                print(
                    f"Record {i} did not match the pattern. Making the segmented text NULL."
                )
            record[f"seg_{add_column_name}"] = "Invalid"
