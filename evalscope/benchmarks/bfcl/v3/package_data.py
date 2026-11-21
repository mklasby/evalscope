"""BFCL v3's test entries from the bfcl-eval package's own data, in the rows the adapter evaluates.

bfcl-eval names the v3 categories as v4 does (simple -> simple_python, java -> simple_java, javascript ->
simple_javascript) and keeps their data in the package. Each row is built as Gorilla's generation and evaluation build
it: the prompt's function docs carry the language hint (``load_dataset_entry``'s default for generation) and the
checker's do not (as ``eval_runner`` loads them). A function-calling model gets the question unchanged and the docs as
OpenAI tools (``convert_to_tool`` in the OpenAI completions style); a prompting model gets Gorilla's system prompt,
merged into the question's first system message as ``system_prompt_pre_processing_chat_model`` does.
"""

from copy import deepcopy
from typing import Any, Dict, List

V3_TO_PACKAGE_CATEGORY = {
    'simple': 'simple_python',
    'java': 'simple_java',
    'javascript': 'simple_javascript',
}

PACKAGE_CATEGORY_LANGUAGE = {
    'simple_java': 'Java',
    'simple_javascript': 'JavaScript',
}


def _openai_tool_functions(function_docs: List[dict]) -> List[dict]:
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool

    return convert_to_tool(function_docs, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)


def load_bfcl_v3_rows(subset: str, is_fc_model: bool) -> List[Dict[str, Any]]:
    """The adapter's rows for one v3 subset (a key of SUBJECT_MAPPING), in the package's entry order."""
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
    from bfcl_eval.utils import (
        is_multi_turn,
        is_relevance_or_irrelevance,
        load_dataset_entry,
        load_ground_truth_entry,
    )

    category = V3_TO_PACKAGE_CATEGORY.get(subset, subset)
    checker_entries = load_dataset_entry(category, include_prereq=False, include_language_specific_hint=False)
    prompt_entries = {
        entry['id']: entry
        for entry in load_dataset_entry(category, include_prereq=False, include_language_specific_hint=True)
    }
    ground_truth = {}
    if not is_relevance_or_irrelevance(category):
        ground_truth = {entry['id']: entry['ground_truth'] for entry in load_ground_truth_entry(category)}

    rows = []
    for checker_entry in checker_entries:
        entry_id = checker_entry['id']
        prompt_entry = prompt_entries[entry_id]
        prompt_functions = prompt_entry['function']
        turns = deepcopy(prompt_entry['question'])
        missed_functions = prompt_entry.get('missed_function', {})
        if is_fc_model:
            tools = _openai_tool_functions(prompt_functions)
            # The adapter appends new_tool[0] of each held-out function as an OpenAI tool.
            missed_functions = {
                turn: [[tool['function']] for tool in _openai_tool_functions(docs)]
                for turn, docs in missed_functions.items()
            }
        else:
            tools = []
            turns[0] = system_prompt_pre_processing_chat_model(turns[0], prompt_functions, entry_id)
        if entry_id not in ground_truth and not is_relevance_or_irrelevance(category):
            raise KeyError(f'BFCL entry {entry_id} has no ground truth.')
        rows.append(
            {
                'id': entry_id,
                'multi_turn': is_multi_turn(category),
                'functions': checker_entry['function'],
                'tools': tools,
                'turns': turns,
                'missing_functions': missed_functions,
                'ground_truth': ground_truth.get(entry_id, {}),
                'initial_config': checker_entry.get('initial_config', {}),
                'involved_classes': checker_entry.get('involved_classes', []),
                'language': PACKAGE_CATEGORY_LANGUAGE.get(category, 'Python'),
                'test_category': subset,
                'subset': subset,
                'should_execute_tool_calls': is_multi_turn(category),
                'is_fc_model': is_fc_model,
            }
        )
    return rows
