# -*- coding: utf-8 -*-
import os
import csv
import argparse
from hydra import initialize, compose
from typing import List
from difflib import SequenceMatcher

def poem_mode(poem: str, mode: int = 2) -> List[str]:
    """
    Function to filter and output data based on the given mode.

    Mode 1: Original output.
    Mode 2: Filter rows where the decomposition code starts with 'A' or 'B' or 'D' and the first digit of the number part is '0'.
    Mode 3: Filter rows where the decomposition code starts with 'A' or 'C' or 'E' and the first digit of the number part is '0',
            and filter out decompositions of multi-sense words (i.e., if a 'B' or 'D' row is found, skip its 'C' or 'E' blocks).

    :param poem: A string block with multiple lines of input data representing the poems.
    :param mode: Mode to determine the filtering behavior:
        - 1: Original output.
        - 2: Basic sense, ignore decomposition ('A', 'B', 'D' with '0' in decomposition code).
        - 3: Basic sense, consider decomposition ('A', 'C', 'E' with '0' in decomposition code), and filter multi-sense decompositions.
    :return: Filtered list of strings based on the mode.
    """
    if mode == 1:
        # If mode is 1, return the original poem string split by lines
        return poem.strip().splitlines()

    # Split the input string by lines and further split each line by whitespace for other modes
    data = [line.split() for line in poem.strip().splitlines()]
    result = []

    skip_decompositions = False  # Flag to skip decompositions related to 'B' or 'D' rows

    for row in data:
        decomposition_code = row[1]  # Second column represents the decomposition code
        first_char = decomposition_code[0]  # Get the first character of the decomposition code (A, B, C, D, or E)

        # Ensure the first character is one of 'A', 'B', 'C', 'D', or 'E'
        assert first_char in ['A', 'B', 'C', 'D', 'E'], f"Unexpected decomposition code: {decomposition_code}"

        # Explanation of decomposition code first_digit:
        # 0: Basic sense (primary meaning)
        # Other integers: Other senses (secondary meanings or further decompositions)
        first_digit = decomposition_code[1]  # Get the first digit of the numeric part (0 for basic sense)

        if mode == 2 and first_char in ['A', 'B', 'D'] and first_digit == '0':
            # Mode 2: Basic sense, ignore decomposition (A, B, D)
            result.append(" ".join(row))
        elif mode == 3:
            if skip_decompositions and first_char in ['C', 'E']:
                # Skip 'C' or 'E' decompositions after 'B' or 'D' multi-sense lines
                continue
            elif first_char in ['B', 'D'] and first_digit != '0':
                # If we encounter a multi-sense row (first digit != '0'), we set the flag to skip its decompositions
                skip_decompositions = True
            else:
                skip_decompositions = False  # Reset the flag if we're not in a multi-sense situation

            if first_char in ['A', 'C', 'E'] and first_digit == '0':
                # Mode 3: Basic sense, consider decomposition (A, C, E)
                result.append(" ".join(row))

    return result


def translation_mode(translation: str, mode: int = 2) -> List[str]:
    """
    Function to filter and output translation data based on the given mode, with an added functionality to number each token
    and remove punctuation based on the POS column (6th column).

    Mode 1: Original output.
        - Returns the translation data with numbering, as is.

    Mode 2: Ambiguity set to 1, ignore decomposition.
        - Filters rows where the first column is 1 (ambiguity set to 1) and the fourth column is 0 or 1 (decomposition ignored),
          and skips the current row if the next row's fourth column is 3.

    Mode 3: Ambiguity set to 1, consider decomposition.
        - Filters rows where the first column is 1 (ambiguity set to 1) and the fourth column is 0, 2, or 3 (decomposition considered),
          and skips the current row if the next row's fourth column is 3.

    Additionally, each token will be numbered based on the decomposition field (4th column) and punctuation will be removed
    based on the POS column (6th column):
        - If the POS column is 76 or greater, the line is considered a punctuation and will be skipped.
        - The number will be added at the end of each line.

    :param translation: A string block with multiple lines of input data, representing the translations.
    :param mode: Mode to determine the filtering behavior.
        - 1: Original output
        - 2: Ambiguity set to 1, ignore decomposition
        - 3: Ambiguity set to 1, consider decomposition
    :return: Filtered list of strings with numbered tokens based on the mode.
    """
    # First, remove punctuation and add global numbering to the tokens
    numbered_data = []
    for line in translation.strip().splitlines():
        clean_line = _remove_punctuation(line)
        if clean_line:  # Skip the line if it's a punctuation
            numbered_data.append(_add_token_numbering(clean_line))

    # If mode is 1, return the data directly without any filtering
    if mode == 1:
        return numbered_data

    # Now apply the mode filtering for mode 2 and 3
    result = []
    for i, row in enumerate(numbered_data):
        row_fields = row.split()
        ambiguity = int(row_fields[0])
        decomposition = int(row_fields[3])

        # Apply filtering based on mode
        if mode == 2 and ambiguity == 1 and decomposition in [0, 1]:
            result.append(row)  # Ambiguity set to 1, ignore decomposition
        elif mode == 3 and ambiguity == 1 and decomposition in [0, 1, 3]:
            # Check if the current row's fourth column is 1, and if the next row's fourth column is 3
            if decomposition == 1 and i + 1 < len(numbered_data) and int(numbered_data[i + 1].split()[3]) == 3:
                continue  # Skip current row if the next row's fourth column is 3

            result.append(row)  # Ambiguity set to 1, consider decomposition

    return result


def _add_token_numbering(line: str) -> str:
    """
    Internal function to add numbering to each line based on the decomposition field (4th column).

    :param line: A single line of the translation data.
    :return: The line with numbering added to the end of the line.
    """
    row = line.split()
    decomposition_field = row[3]  # The decomposition field (4th column)

    # Static variable to hold the token counter across function calls
    if not hasattr(_add_token_numbering, "token_counter"):
        _add_token_numbering.token_counter = 0

    # Check the decomposition field and update the token counter accordingly
    if decomposition_field != "0" and decomposition_field != "1":
        row.append(str(_add_token_numbering.token_counter))
    else:
        _add_token_numbering.token_counter += 1
        row.append(str(_add_token_numbering.token_counter))

    return " ".join(row)


def _remove_punctuation(line: str) -> str:
    """
    Internal function to remove lines that are considered punctuation based on the POS column (6th column).

    :param line: A single line of the translation data.
    :return: The original line if it is not punctuation, otherwise an empty string.
    """
    row = line.split()
    pos_column = int(row[4])  # POS column is the 5th column
    polysemy_colomn = row[0]  # Polysemy coloum is first column

    # POS 76 and greater are considered punctuation, so we skip them
    # When polysemy_colomn is N, the row is un validated, so we skip them
    if pos_column >= 76 or polysemy_colomn == "N":
        return ""  # Return an empty string to indicate this line is punctuation and should be skipped

    return line


def LCS(s: str, t: str) -> int:
    """
    Calculate the length of the longest common subsequence (LCS) between two strings.

    This is an internal function that uses a sequence matching algorithm to determine
    the longest common subsequence between the two input strings.

    :param s: The first input string.
    :param t: The second input string.
    :return: The length of the longest common subsequence between the two strings.
    """
    seq_matcher = SequenceMatcher(None, s, t)
    match = seq_matcher.find_longest_match(0, len(s), 0, len(t))
    return match.size


def match_category(s: str, t: str) -> str:
    """
    Classify the match between two strings into one of four categories based on the LCS (Longest Common Subsequence) length.

    The function calculates the LCS length between two strings and classifies the match into one of four categories:
    - 'U': [U]nmatch, when LCS length is less than 10.
    - 'G': [G]roup match, when LCS length is between 10 and 12.
    - 'F': [F]ield match, when LCS length is between 13 and 16.
    - 'E': [E]xact match, when LCS length is 17 or greater.

    :param s: The first input string to compare.
    :param t: The second input string to compare.
    :return: A string representing the match category ('U', 'G', 'F', or 'E') based on the LCS length.
    """
    # Calculate LCS length using the internal function
    lcs_length = LCS(s, t)

    # Categorize based on the LCS length
    if lcs_length < 10:
        return "U"
    elif 10 <= lcs_length < 13:
        return "G"
    elif 13 <= lcs_length < 17:
        return "F"
    else:
        return "E"


def match_count_bag(poem_lines: List[str], translation_lines: List[str]) -> dict:
    """
    Calculate the total match count between the poem and translation for Exact (E), Field (F), and Group (G) matches.

    The function compares each element in the poem with the elements in the translation.
    The match is counted with priority: Exact (E) > Field (F) > Group (G), meaning if an E match is found,
    it will not check for F or G, and similarly for F before G.

    :param poem_lines: A list of strings representing the poem data (already filtered).
    :param translation_lines: A list of strings representing the translation data (already filtered).

    :return: A dictionary containing the total match counts for 'E', 'F', and 'G' categories.

    Example Usage:
    --------------
    poem_data = [
        "01:000001:0001 A00 BG-01-1630-01-0100 02 年 年 とし 年 とし",
        "01:000001:0002 A00 BG-08-0061-07-0100 61 の の の の の"
    ]

    translation_data = [
        "1 katagiri 0001 1 51 50 07 BG-03-1940-01-010-A 早く はやい 早い 1",
        "1 katagiri 0001 2 51 50 07 BG-03-1660-03-010-A -- はやい 早い 1"
    ]

    match_counts = match_count(poem_data, translation_data)
    print(f"Match counts: {match_counts}")
    """

    # Initialize counters for E, F, G matches
    match_counts = {"E": 0, "F": 0, "G": 0}

    # Calculate total match count based on poem as the source
    for poem_line in poem_lines:
        s = poem_line.split()[2]  # BG ID string from poem
        assert len(s) == 18, f"Invalid BG ID string: {s}"

        found_match = False

        # First, search for E match
        for translation_line in translation_lines:
            t = translation_line.split()[7]  # BG ID string from translation
            assert len(t) == 19, f"Invalid BG ID string: {t}"

            if match_category(s, t) == "E":
                match_counts["E"] += 1
                found_match = True
                break  # Stop once an E match is found

        # If no E match is found, search for F match
        if not found_match:
            for translation_line in translation_lines:
                t = translation_line.split()[7]
                if match_category(s, t) == "F":
                    match_counts["F"] += 1
                    found_match = True
                    break  # Stop once an F match is found

        # If no F match is found, search for G match
        if not found_match:
            for translation_line in translation_lines:
                t = translation_line.split()[7]
                if match_category(s, t) == "G":
                    match_counts["G"] += 1
                    break  # Stop once a G match is found

    # Assert to ensure that the total matches (E + F + G) equal the number of poem lines
    total_matches = match_counts["E"] + match_counts["F"] + match_counts["G"]
    assert total_matches <= len(poem_lines), f"Total matches {total_matches} exceed the number of poem lines {len(poem_lines)}"

    return match_counts


def weight(s: str, t: str, u: int = -1, g: int = 10, f: int = 13, e: int = 17) -> int:
    """
    Calculate the weight based on the match category between two strings.

    The function first classifies the match category using the `match_category` function.
    It allows custom weight values for each match category (U, G, F, E).
    - 'U' (Unmatch) returns the weight for unmatch (default is -1).
    - 'G' (Group match) returns the weight for group match (default is 10).
    - 'F' (Field match) returns the weight for field match (default is 13).
    - 'E' (Exact match) returns the weight for exact match (default is 17).

    :param s: The first input string to compare.
    :param t: The second input string to compare.
    :param u: Weight for unmatch (default is -1).
    :param g: Weight for group match (default is 10).
    :param f: Weight for field match (default is 13).
    :param e: Weight for exact match (default is 17).
    :return: An integer weight based on the match category.
    """
    category = match_category(s, t)

    if category == "U":
        return u
    elif category == "G":
        return g
    elif category == "F":
        return f
    else:  # "E"
        return e


def alignment(poem_lines: List[str], translation_lines: List[str], gap_penalty: float = 0.01, u: int = -1, g: int = 10, f: int = 13, e: int = 17) -> str:
    """
    Align the poem and translation sequences using dynamic programming and return the alignment in a formatted output.

    :param poem_lines: A list of strings representing the poem data (already filtered).
    :param translation_lines: A list of strings representing the translation data (already filtered).
    :param gap_penalty: Float representing the penalty for inserting gaps (default is 0.01).
    :param u: Weight for unmatch (default is -1).
    :param g: Weight for group match (default is 10).
    :param f: Weight for field match (default is 13).
    :param e: Weight for exact match (default is 17).

    :return: A formatted string representing the aligned sequences, including matching category and token information.
    """

    # Get the size of the poem and translation
    poem_size = len(poem_lines)
    translation_size = len(translation_lines)

    # m: number of poem lines, n: number of translation lines
    m, n = poem_size, translation_size

    # Initialize the DP (Dynamic Programming) table and traceback table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    traceback = [[(0, 0)] * (n + 1) for _ in range(m + 1)]

    # Fill DP table with gap penalties for alignment
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty  # Penalty for gaps in translation
        traceback[i][0] = (i - 1, 0)  # Record traceback
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty  # Penalty for gaps in poem
        traceback[0][j] = (0, j - 1)  # Record traceback

    # Fill the DP table with alignment scores based on the weight function
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_weight = dp[i - 1][j - 1] + weight(poem_lines[i - 1], translation_lines[j - 1], u=u, g=g, f=f, e=e)
            gap_poem = dp[i - 1][j] + gap_penalty  # Gap in translation
            gap_translation = dp[i][j - 1] + gap_penalty  # Gap in poem

            dp[i][j] = max(match_weight, gap_poem, gap_translation)

            # Track the source of the best alignment decision (match, gap in poem, gap in translation)
            if dp[i][j] == match_weight:
                traceback[i][j] = (i - 1, j - 1)
            elif dp[i][j] == gap_poem:
                traceback[i][j] = (i - 1, j)
            else:
                traceback[i][j] = (i, j - 1)

    # Traceback step to retrieve the optimal alignment path
    aligned_poem = []
    aligned_translation = []

    i, j = m, n
    while i > 0 or j > 0:
        prev_i, prev_j = traceback[i][j]
        if i > 0 and j > 0 and (prev_i, prev_j) == (i - 1, j - 1):
            aligned_poem.append(poem_lines[i - 1])
            aligned_translation.append(translation_lines[j - 1])
        elif i > 0 and (prev_i, prev_j) == (i - 1, j):
            aligned_poem.append(poem_lines[i - 1])
            aligned_translation.append("-")  # Gap in translation
        else:
            aligned_poem.append("-")  # Gap in poem
            aligned_translation.append(translation_lines[j - 1])
        i, j = prev_i, prev_j

    # Reverse the alignments to reflect the original sequence
    aligned_poem.reverse()
    aligned_translation.reverse()

    # Initialize output
    output = []

    # Process each aligned pair for final output
    for pair_id, (op_token, ct_token) in enumerate(zip(aligned_poem, aligned_translation), 1):

        # Extract necessary fields for poem and translation tokens
        op_fields = op_token.split() if op_token != "-" else ["-"] * 9  # Handle gaps with placeholder
        ct_fields = ct_token.split() if ct_token != "-" else ["-"] * 12  # Handle gaps with placeholder

        assert len(op_fields) == 9, f"Invalid op_fields length: {len(op_fields)}. Fields: {op_fields}"
        
        # Unpack poem fields
        (
            token_identifier_op,  # AnthologyID:PoemID:SequentialID
            polysemy_decomposition_op,
            bg_id_op,
            pos_op,
            surface_op,
            lemma_kanji_op,
            lemma_kana_op,
            conjugation_kanji_op,
            conjugation_kana_op
        ) = op_fields

        # Unpack translation fields
        (
            polysemy_ct,
            # Explanation for polysemy_ct:
            # 1=non-polysemy
            # 2=polysemy
            translator,
            poem_id,
            polysemy_decomposition_ct,
            # Explanation for polysemy_decomposition_ct:
            # 0=default sense(non-polysemy and simplex);
            # 1=defaut sense(compound);
            # 2=potential sense; 3=decomposition of compound
            pos_ct,
            pos_b_ct,
            pos_c_ct,
            bg_id_ct,
            surface_ct,
            lemma_kana_ct,
            lemma_kanji_ct,
            seq_id_ct
        ) = ct_fields

        # Handle potential sense and decomposition flags
        potential_sense_op = "" if (polysemy_decomposition_op == "-" or polysemy_decomposition_op[1] == "0") else "*"
        potential_sense_ct = "" if (polysemy_decomposition_ct == "-" or (polysemy_ct == "1" and polysemy_decomposition_ct != "2")) else "*"
        decomposition_op = "+" if (polysemy_decomposition_op[0] == "C" or polysemy_decomposition_op[0] == "E") else ""
        decomposition_ct = "+" if polysemy_decomposition_ct == "3" else ""

        # Determine match category based on score (Exact, Field, Group)
        category = match_category(bg_id_op, bg_id_ct) if (op_token != "-" and ct_token != "-") else "-"
        match_value = category

        # Sequential ID for poem
        seq_id_op = int(token_identifier_op.split(":")[-1]) if token_identifier_op != "-" else "-"

        # Padding
        padding_width_op = 7 - 1 * len(surface_op) if surface_op != "-" else 7
        if padding_width_op < 0:
            padding_width_op = 0
        padding_width_ct = 7 - 1 * len(lemma_kanji_ct) if lemma_kanji_ct != "-" else 7
        if padding_width_ct < 0:
            padding_width_ct = 0

        # Format the final aligned output with appropriate columns and alignment
        output.append(
            f"{pair_id:>2} {match_value:>2} {pos_op:>2} {bg_id_op:>18} {surface_op:>{padding_width_op}} {seq_id_op:>2} {potential_sense_op:>1} "
            f"{decomposition_op:>1} <-> {decomposition_ct:<1} {potential_sense_ct:<1} {seq_id_ct:<2} {lemma_kanji_ct:<{padding_width_ct}} {bg_id_ct:<18}"
        )

    # Prepare the formatted output with headers and alignment
    translator = translation_lines[0].split()[1]
    poem_id = translation_lines[0].split()[2]
    header = (
        f"args: translator:{translator}; poem No.{poem_id}; weight:(U={u}, G={g}, F={f}, E={e}); gap penalty: {gap_penalty}\n"
        " +------- pair No.\n"
        " |  +---- matching level (E=Exact, F=Field, G=Group)\n"
        " |  |  +- pos No.\n"
        " |  |  |  OP decomposition (+) ----------+     +--------------- CT decomposition\n"
        " |  |  |  OP potential sense (*) ------+ |     | +------------- CT potential sense\n"
        " |  |  |  OP token No. --------------+ | |     | | +----------- CT token No.\n"
        " |  |  |  OP token ---------------+  | | |     | | |  +-------- CT token\n"
        " |  |  |  OP WLSP code ---+       |  | | |     | | |  |       + CT WLSP code\n"
        " |  |  |                  |       |  | | |     | | |  |       |"
    )
    output = [header] + output

    return "\n".join(output)


def match_count_alignment(alignment_output: str) -> dict:
    """
    Count the occurrences of match categories (E, F, G) in the alignment output.

    This function counts only the E (Exact), F (Field), and G (Group) matches in the alignment output.
    It does not calculate or track unmatched (U) tokens.

    :param alignment_output: A string representing the alignment output, where each line represents a pair.

    :return: A dictionary with counts of E, F, and G matches.
    """
    # Initialize the counters for each match category
    counts = {"E": 0, "F": 0, "G": 0}

    # Extract the total number of OP tokens from the header
    lines = alignment_output.strip().splitlines()

    # Iterate through the lines and count occurrences of E, F, G
    for line in lines:
        # Skip lines that do not contain match category information
        if len(line.strip()) == 0 or line.strip().startswith('+'):
            continue

        # Extract the match category from the line (2nd field)
        match_category = line.split()[1].strip()

        # Count E, F, G and skip lines with "-"
        if match_category == "E":
            counts["E"] += 1
        elif match_category == "F":
            counts["F"] += 1
        elif match_category == "G":
            counts["G"] += 1

    return counts


def match_count(match_counts: dict, level: int = 1) -> int:
    """
    Calculate the total match count based on the specified match level.

    The match count is calculated by summing the appropriate categories of matches
    based on the selected level of strictness:

    - Level 1: Only count Exact (E) matches.
    - Level 2: Count Exact (E) and Field (F) matches.
    - Level 3: Count Exact (E), Field (F), and Group (G) matches.

    :param match_counts: A dictionary with counts of E, F, G, and U matches.
    :param level: Integer representing the matching strictness level (default is 1).
        - Level 1: Only count Exact matches (E).
        - Level 2: Count Exact (E) and Field (F) matches.
        - Level 3: Count Exact (E), Field (F), and Group (G) matches.

    :return: An integer representing the total match count.
    """

    # Ensure the level is valid
    assert level in [1, 2, 3], "Invalid level. Must be 1, 2, or 3."

    # Calculate the total match count based on the level
    if level == 1:
        total_match_count = match_counts["E"]
    elif level == 2:
        total_match_count = match_counts["E"] + match_counts["F"]
    else:  # level == 3
        total_match_count = match_counts["E"] + match_counts["F"] + match_counts["G"]

    return total_match_count


def main(poem: str, translation: str, mode: int = 3, level: int = 3, gap_penalty: float = 0.01, u: int = -1, g: int = 10, f: int = 13, e: int = 17, output_folder: str = "output"):
    """
    A complete pipeline function that processes poem and translation data, computes match counts,
    alignment results, and rates (addition and unmatch rates), and writes results to a CSV file.

    :param poem: The poem data as a string.
    :param translation: The translation data as a string.
    :param mode: The mode to filter the poem and translation (default is 3).
    :param level: The strictness level for calculating match counts (default is 3).
    :param gap_penalty: Gap penalty for alignment (default is 0.01).
    :param u: Weight for unmatch (default is -1).
    :param g: Weight for group match (default is 10).
    :param f: Weight for field match (default is 13).
    :param e: Weight for exact match (default is 17).
    :param output_folder: The folder where the CSV file will be saved (default is "output").
    """

    # Process the poem and translation using poem_mode and translation_mode
    poem_lines = poem_mode(poem, mode)
    translation_lines = translation_mode(translation, mode)

    # Calculate poem and translation sizes
    poem_size = len(poem_lines)
    translation_size = len(translation_lines)

    # Info
    translator = translation_lines[0].split()[1]
    poem_id = translation_lines[0].split()[2]

    # Calculate match counts using the bag method
    match_count_bag_ = match_count_bag(poem_lines, translation_lines)
    exact_match_bag = match_count_bag_["E"]
    field_match_bag = match_count_bag_["F"]
    group_match_bag = match_count_bag_["G"]
    unmatch_bag = poem_size - exact_match_bag - field_match_bag - group_match_bag
    total_match_bag = match_count(match_count_bag_, level)

    # Calculate alignment and match counts from alignment
    alignment_output = alignment(poem_lines, translation_lines, gap_penalty=gap_penalty, u=u, g=g, f=f, e=e)
    match_count_alignment_ = match_count_alignment(alignment_output)
    exact_match_alignment = match_count_alignment_["E"]
    field_match_alignment = match_count_alignment_["F"]
    group_match_alignment = match_count_alignment_["G"]
    unmatch_alignment = poem_size - exact_match_alignment - field_match_alignment - group_match_alignment
    total_match_alignment = match_count(match_count_alignment_, level)

    # Calculate addition and unmatch rates
    addition_rate_bag = 1 - total_match_bag / translation_size if translation_size > 0 else 0
    addition_rate_alignment = 1 - total_match_alignment / translation_size if translation_size > 0 else 0
    unmatch_rate_bag = unmatch_bag / poem_size if poem_size > 0 else 0
    unmatch_rate_alignment = unmatch_alignment / poem_size if poem_size > 0 else 0

    # Format the statistics according to the required structure
    statistics = (
        f"mode={mode}; level={level}\n"
        f"OP={poem_size}; CT={translation_size};\n"
        f"bag (E={exact_match_bag}, F={field_match_bag}, G={group_match_bag}, U={unmatch_bag}, T={total_match_bag}, "
        f"AddRate={addition_rate_bag:.2%}, UnmatchRate={unmatch_rate_bag:.2%});\n"
        f"alignment (E={exact_match_alignment}, F={field_match_alignment}, G={group_match_alignment}, U={unmatch_alignment}, "
        f"T={total_match_alignment}, AddRate={addition_rate_alignment:.2%}, UnmatchRate={unmatch_rate_alignment:.2%})"
    )

    # Prepare the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Prepare CSV file name and path
    csv_filename = f"{poem_id}-{translator}.csv"
    csv_filepath = os.path.join(output_folder, csv_filename)

    # Write results to CSV
    with open(csv_filepath, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write CSV header
        csvwriter.writerow([
            "translator", "poem_id", "poem_size", "translation_size", "total_match_bag", "unmatch_bag",
            "exact_match_bag", "field_match_bag", "group_match_bag", "addition_rate_bag", "unmatch_rate_bag",
            "total_match_alignment", "unmatch_alignment", "exact_match_alignment", "field_match_alignment",
            "group_match_alignment", "addition_rate_alignment", "unmatch_rate_alignment"
        ])
        # Write CSV row
        csvwriter.writerow([
            translator, poem_id, poem_size, translation_size, total_match_bag, unmatch_bag, exact_match_bag,
            field_match_bag, group_match_bag, addition_rate_bag, unmatch_rate_bag, total_match_alignment,
            unmatch_alignment, exact_match_alignment, field_match_alignment, group_match_alignment,
            addition_rate_alignment, unmatch_rate_alignment
        ])

    print(f"output file: {csv_filepath}")
    # Print alignment output and formatted statistics
    print("statistics:", statistics)
    print(alignment_output)


def cli_main():
    """
    The CLI main function for handling arguments and calling the main logic.
    """
    # Setup argument parser for poem and translation inputs
    parser = argparse.ArgumentParser(description='Process poem and translation data.')
    parser.add_argument('--poem', type=str, required=True, help='The poem text input.')
    parser.add_argument('--translation', type=str, required=True, help='The translation text input.')

    # Parse arguments
    args = parser.parse_args()

    # Use Hydra to initialize and compose configuration for additional parameters
    with initialize(version_base=None, config_path="../parameters", job_name=""):
        cfg = compose(config_name="default")

    # Call the main function with both argparse and Hydra parameters
    main(poem=args.poem, translation=args.translation, mode=cfg.mode, level=cfg.level, gap_penalty=cfg.gap_penalty,
         u=cfg.u, g=cfg.g, f=cfg.f, e=cfg.e, output_folder=cfg.output_folder)


if __name__ == "__main__":
    cli_main()
