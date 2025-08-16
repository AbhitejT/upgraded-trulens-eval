from trulens_eval.feedback import GroundTruthAgreement
from trulens_eval import Tru
from trulens_eval import OpenAI, Feedback, TruLlama
import numpy as np
import pandas as pd
import json
from pathlib import Path

from generate import generate_qa_pairs_from_documents

# Initialize TruLens
tru = Tru()
openai_provider = OpenAI(model_engine="gpt-3.5-turbo-1106")

def load_qa_pairs_from_generated_data(file_path: str = "test_data.jsonl"):
    # Load QA pairs from generated JSONL file
    qa_set = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                qa_pair = json.loads(line.strip())
                qa_set.append({
                    "query": qa_pair["question"],
                    "response": qa_pair["truth"],
                    "expected_response": qa_pair["truth"]
                })
        print(f"Loaded {len(qa_set)} QA pairs from {file_path}")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using empty QA set.")
        qa_set = []
    return qa_set

def ensure_qa_pairs_exist(
    document_folder: str = "./your_documents",
    output_file: str = "test_data.jsonl",
    force_regenerate: bool = False
):
    # Generate QA pairs if they don't exist
    if force_regenerate or not Path(output_file).exists():
        if not Path(document_folder).exists():
            print(f"Warning: Document folder '{document_folder}' does not exist.")
            print("Please create this folder and add your documents, or specify a different folder.")
            print("For now, falling back to CSV data...")
            return False
            
        print("Generating QA pairs...")
        try:
            generate_qa_pairs_from_documents(
                document_folder=document_folder,
                output_file=output_file,
                num_questions_total=200,
                num_questions_per_source=5
            )
            return True
        except Exception as e:
            print(f"Error generating QA pairs: {e}")
            print("Falling back to CSV data...")
            return False
    else:
        print(f"Using existing QA pairs from {output_file}")
        return True

def get_qa_set(use_generated: bool = True, csv_file: str = "ipcc_test_questions.csv"):
    # Get QA pairs from either generated data or CSV file
    if use_generated:
        success = ensure_qa_pairs_exist()
        if success:
            qa_set = load_qa_pairs_from_generated_data()
            if not qa_set:
                print("Generated data is empty, falling back to CSV data...")
                return get_qa_set(use_generated=False, csv_file=csv_file)
            return qa_set
        else:
            print("QA generation failed, falling back to CSV data...")
            return get_qa_set(use_generated=False, csv_file=csv_file)
    else:
        # Load from CSV (original method)
        qa_df = pd.read_csv(csv_file)
        qa_set = [{"query": item["Question"], "response": item["Answer"], "expected_response": item["Answer"]} for index, item in qa_df.iterrows()]
        return qa_set

# Initialize as None, will be loaded when needed
qa_set = None

def get_qa_set_lazy(use_generated: bool = True, csv_file: str = "ipcc_test_questions.csv"):
    # Get QA set, loading it if not already loaded
    global qa_set
    if qa_set is None:
        qa_set = get_qa_set(use_generated=use_generated, csv_file=csv_file)
    return qa_set

# Define evaluation metrics
f_qa_relevance = Feedback(
    openai_provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()

f_qs_relevance = Feedback(
    openai_provider.relevance_with_cot_reasons, name="Context Relevance"
).on_input().on(TruLlama.select_source_nodes().node.text).aggregate(np.mean)

f_groundedness = (
    Feedback(openai_provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(TruLlama.select_source_nodes().node.text)
    .on_output()
    .aggregate(np.mean)
)

# All metrics for comprehensive evaluation
metrics = [f_qa_relevance, f_qs_relevance, f_groundedness]

def get_metrics():
    # Get metrics list, adding groundtruth when needed
    global metrics, qa_set
    if qa_set is None:
        qa_set = get_qa_set_lazy(use_generated=False)
    current_metrics = metrics.copy()
    if qa_set and len(qa_set) > 0:
        try:
            gta = GroundTruthAgreement(ground_truth=qa_set, provider=openai_provider)
            f_answer_correctness = (
                Feedback(gta.agreement_measure, name="Answer Correctness")
                .on_input_output()
            )
            current_metrics.append(f_answer_correctness)
        except Exception as e:
            print(f"Warning: Could not set up Answer Correctness: {e}")
    return current_metrics

def get_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        feedbacks=get_metrics(),
        app_id=app_id
    )
    return tru_recorder 