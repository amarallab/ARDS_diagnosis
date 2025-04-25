"""
This module provides a collection of utility functions for processing and cleaning text data, 
specifically tailored for medical reports such as chest X-ray reports and notes.
The functions include text normalization, section removal, stopword elimination, stemming,
and extraction of contextual information around specific risk factors.
These tools are designed to facilitate the preprocessing of medical text data for downstream
tasks such as machine learning and natural language processing.

Author(s): Luís A. Nunes Amaral, Félix L. Morales
"""
import re
from operator import itemgetter
from pathlib import Path


def process_times(note: str, pattern: str) -> str:
    """
    Look for the presence of times in a given text, either in 12-hour or 24-hour clock format,
    using a regular expression pattern, and replace them with the format "aahbb".

    Args:
        note (str): The input string containing the text to process.
        pattern (str): The regular expression pattern to match time formats.

    Returns:
        str: The processed string with times replaced by the "aahbb" format.
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


def clean_up_string(text: str) -> str:
    """
    Cleans up issues in a given text, such as missing spaces, commas, and formatting inconsistencies.

    Args:
        text (str): The input string containing the text to clean.

    Returns:
        str: The cleaned and formatted text.
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

    # Add a space after a colon, remove parenthesis and brackets, and
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


def remove_easy_sections(report: str, section_order: list[tuple[str, bool]], verbose: bool = False) -> str:
    """
    Removes specified sections from a chest X-ray report based on section names followed by a colon.

    Args:
        report (str): The input string containing the chest X-ray report.
        section_order (list[tuple[str, bool]]): A list of tuples where each tuple contains a section name (str) 
                                                and a boolean indicating whether to keep (True) or remove (False) 
                                                the section.
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        str: The processed chest X-ray report with specified sections removed.
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


def handle_subsection_titles(report: str) -> list[str]:
    """
    Processes a chest X-ray report to identify lines with subsection titles,
    splits them at colons, and reconnects the parts into a structured format.

    Args:
        report (str): A string containing the chest X-ray report.

    Returns:
        list[str]: A list of strings where subsection titles are split and formatted.
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


def remove_lines_on_other_organs(report: list[str], exclusion_set: set[str]) -> list[str]:
    """
    Removes statements from a report that mention terms indicative of organs other than lungs.

    Args:
        report (list[str]): A list of statements (strings) from the report.
        exclusion_set (set[str]): A set of terms to exclude, representing organs other than lungs.

    Returns:
        list[str]: A new list of statements with excluded terms removed.
    """
    new_report = []

    for line in report:
        if ":" in line:
            new_report.append(line)
        else:
            if not exclusion_set.intersection(set(line.split())):
                new_report.append(line)

    return new_report


def curate_indicator_word_list(
    filename: str, targeted_stemming: dict[str, str], verbose: bool = False
) -> list[str]:
    """
    Reads a file containing raw indicator words of bilateral infiltrates in chest X-rays,
    processes them, and returns a cleaned and stemmed list of such words.

    Args:
        filename (str): The name of the file (without extension) containing raw indicator words.
        targeted_stemming (dict[str, str]): A dictionary with specific key-value pairs for stemming.
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        list[str]: A sorted list of cleaned and stemmed indicator words.
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


def stem_indicator_words(report: list[str], targeted_stemming: dict[str, str]) -> list[str]:
    """
    Stems indicator words in the given report based on a targeted stemming dictionary.

    Args:
        report (list[str]): A list of strings representing the report.
        targeted_stemming (dict[str, str]): A dictionary where keys are words to be stemmed 
                                            and values are their stemmed replacements.

    Returns:
        list[str]: A new list of strings with indicator words stemmed according to the dictionary.
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


def remove_stopwords(
    report: list[str], complex_stopwords: list[str], simple_stopwords: list[str]
) -> list[str]:
    """
    Removes both complex and simple stopwords from the given report.

    Args:
        report (list[str]): A list of strings representing the report.
        complex_stopwords (list[str]): A list of phrases or substrings to be removed from the report.
        simple_stopwords (list[str]): A list of individual words to be removed from the report.

    Returns:
        list[str]: A new list of strings with stopwords removed.
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


def remove_sections_n_duplicate_lines(report: list[str]) -> list[str]:
    """
    Removes duplicate statements and section titles from a report.

    Args:
        report (list[str]): A list of strings representing the report.

    Returns:
        list[str]: A new list of strings with duplicates and section titles removed.
    """
    new_report = []

    for line in report:
        if line not in new_report and ":" not in line:
            new_report.append(line)

    return new_report


def refine_cleaning(report: list[str], useless_statements: list[str]) -> list[str]:
    """
    Removes useless and numeric statements from a report.

    Args:
        report (list[str]): A list of strings representing the report.
        useless_statements (list[str]): A list of strings considered useless and to be removed.

    Returns:
        list[str]: A new list of strings with useless and numeric statements removed.
    """
    new_report = []

    for line in report:
        logic = False
        if not line.isnumeric():
            # Check if at least one useless statement matches the line.
            for item in useless_statements:
                if line.strip() == item.strip():
                    logic = True
                    break
        else:
            logic = True

        if not logic:
            new_report.append(line)

    return new_report


def remove_dictation(report: list[str], dictation_string: str, verbose: bool = False) -> list[str]:
    """
    Removes lines from a report that start with a specified dictation string.

    Args:
        report (list[str]): A list of strings representing the report.
        dictation_string (str): The string to check at the start of each line.
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        list[str]: A new list of strings with lines starting with the dictation string removed.
    """
    new_report = []

    for line in report:
        if line.startswith(dictation_string):
            if verbose:
                print("Line starts with the dictation string.")
        else:
            if verbose:
                print("Line does not start with the dictation string.")
            new_report.append(line)

    return new_report


def extract_surroundings_of_risk_factor_and_process(
    risk_factor_label: list[dict],
    text_field: str = None,
    add_column_name: str = None,
    verbose: bool = False,
) -> None:
    """
    Extracts and processes text surrounding mentions of specified risk factors in a dataset.

    Args:
        risk_factor_label (list[dict]): A list of dictionaries containing notes data with flags
                                        for a given risk factor.
        text_field (str): The name of the column/field containing the note text.
        add_column_name (str): The risk factor name used for naming the new column.
                               The new column will follow the pattern: "seg_{add_column_name}".
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        None: The function modifies the input `risk_factor_label` in place by adding a new column
              with the extracted and processed text surrounding the risk factor mentions.
    """

    # TODO: Send these to a file
    # Patterns to search for
    patterns = {
        'pneumonia': r"(?<!\w)(?:PCPpneumonia|pneumonia|Pneumonia|PNEUMONIA|pneumoniae|pneunonia|pneunoniae|pnuemonia|bronchopneumonia|parapneumonic|PNA|CAP|VAP|HAP|HCAP|hcap|infection|abx|PCP)(?!\w)",
        'aspiration': r"(?i)(?<!\w)(?<!possibility\sof\s)(?<!\(\?)(?<!no\s{4}e\/o\s)(?<!unclear\sif\sthis\sis\s)(?<!cannot\srule\sout\s)(?<!risk\sfor\s)(?<!risk\sof\s)(?<!\?\s)(?<!cover\sfor\s)(?<!no\switnessed\s)(?:aspiration|aspirating)(?!\svs)(?!\svs.)(?!\?)(?!\ss\/p\sR\smainstem\sintubation)(?!\sprecautions)(?!\sand\sdrainage)(?!\w)",
        'inhalation': r"(?i)(?<!\w)(?:inhaled|inhalation)(?!\w)",
        'pulm_contusion': r"(?i)(?<!\w)(?:pulmonary|pulmoanry)\s+(?:contusion|contusions)(?!\w)",
        'vasculitis': r"(?i)(?<!\w)(?<!\?\s)(?<!less\slikely\s)(?:pulmonary\svasculitis|vasculitis)(?!\slabs)(?!\sworkup)(?!\sand\scarcinomatosis\sis\sless\slikely)(?!\shighly\sunlikely)(?!\sless\slikely)(?!\w)",
        'drowning': r"(?i)(?<!\w)(?:drowned|drowning)(?!\w)",
        'sepsis': r"(?i)(?<!\w)(?:sepsis|urosepsis|septiuc|septic|ssepsis|sseptic|spetic)(?!\w)",
        'shock': r"(?i)(?<!\w)(?:shock|shocks|schock)(?!\w)",
        'overdose': r"(?i)(?<!\w)(?:overdose|drug\soverdose)(?!\w)",
        'trauma': r"(?i)(?<!\w)(?<!OGT\s)(?:trauma|traumatic|barotrauma|barotraumatic)(?!\w)",
        'pancreatitis': r"(?i)(?<!\w)pancreatitis(?!\w)",
        'burn' : r"(?i)(?<!\w)(?:burn|burns)(?!\w)",
        'trali': r"(?<!\w)(?:TRALI|transfusion(?:-|\s)related\sacute\slung\sinjury|transfusion(?:-|\s)associated\sacute\slung\sinjury)(?!\w)",
        'ards': r"(?i)(?<!\w)(?:ards|acute\srespiratory\sdistress\ssyndrome|acute\slung\sinjury|ali|ardsnet|acute\shypoxemic\srespiratory\sfailure)(?!\w)",
        'pregnant': r"(?i)(?<!\w)(?:IUP|G\dP\d)(?!\w)",
        'chf': r"(?i)(?<!\w)(?<!h\/o\s)(?:congestive\sheart\sfailure|chf|diastolic\sHF|systolic\sHF|heart\sfailure|diastolic\sdysfunction|LV\sdysfunction|low\scardiac\soutput\ssyndrome|low\scardiac\soutput\ssyndrom|low\scardiac\souput\ssyndrome|low\sCO\sstate)(?!\swith\spreserved\sef)(?!\swas\sanother\spossible\sexplan)(?!\w)",
        'cardiogenic': r"(?i)(?<!\w)(?<!no\se\/o\sobstructive\sor\s)(?<!versus\s)(?<!rule\sout\s)(?<!ruled\sout\s)(?<!less\slikley\s)(?<!w\/o\sevidence\ssuggestive\sof\s\s)(?<!non\s)(?<!less\slikely\s)(?<!not\slikely\s)(?<!unlikely\sto\sbe\s)(?<!no\sclear\sevidence\sof\sacute\s)(?<!non-)(?<!than\s)(?<!no\sevidence\sof\s)(?:cardiogenic|cardigenic|cardiogemic|cardiac\spulmonary\sedema|cardiac\sand\sseptic\sshock|Shock.{1,15}suspect.{1,15}RV\sfailure)(?!\s\(not\slikely\sgiven\sECHO\sresults\))(?!\sshock\sunlikely)(?!\svs\.\sseptic)(?!\scomponent\salthough\sSvO2\snormal)(?!\w)",
        'non_cardiogenic': r"(?i)(?<!\w)(?:non(?:-|\s)cardiogenic|noncardiogenic|non(?:-|\s)cardigenic|noncardigenic)(?!\w)",
        'palliative': r"(?i)(?<!\w)(?:palliative\scare|comfort\scare|withdraw\scare|comfort\salone|withdraw\ssupport\sin\sfavor\sof\spalliation)(?!\w)",
        'cardiac_arrest': r"(?i)(?<!\w)(?:arrest|cardiorespiratory\sarrest)(?!\w)",
        'dementia': r"(?i)(?<!\w)dementia(?!\w)",
        'stroke': r"(?i)(?<!\w)(?:stroke|strokes|cerebellar\shemorrhage|intracerebral\shemorrhage|BG\shemorrhage|cva|cerebrovascular\saccident|cefrebellar\sinfarcts\/basilar\sstenosis)(?!\w)",
        'alcohol': r"(?i)(?<!\w)(?:alcohol\swithdrawal|dts|dt''s|dt|alcohol\sdependence|alcohol\sabuse|etoh\sabuse|etoh\swithdrawal|etoh\swithdrawl|etoh\sw\/drawal|delirium\stremens)(?!\w)",
        'seizure': r"(?i)(?<!\w)(?<!no\se/o\ssubclinical\s)(?<!no\se/o\ssubclinical\s)(?:seizure|seizures)(?!\w)",
        'ami': r"(?i)(?<!\w)(?:ami|acute\smyocardial\sischemia|acute\smyocardial\sinfarction|myocardial\sinfarction|nstemi|non-st\selevation\smi|stemi|st\selevation\smi|acute\smi)(?!\w)"
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
