import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

@dataclass
class QasperExample:
    id: str
    title: str
    context: str  # Full paper text
    question: str
    ground_truth: str
    metadata: Dict[str, Any]

class QasperDatasetLoader:
    def __init__(self, split: str = "test"):
        self.split = split
        self.dataset = None

    def load(self) -> List[QasperExample]:
        """Loads and processes the QASPER dataset."""
        if load_dataset is None:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading QASPER dataset (split={self.split})...")
        # Removing trust_remote_code as suggested by warning, though it might be needed for some datasets. 
        # Using it if it works, otherwise relying on cache.
        try:
            self.dataset = load_dataset("allenai/qasper", split=self.split, trust_remote_code=True)
        except Exception:
            self.dataset = load_dataset("allenai/qasper", split=self.split)
        
        examples = []
        for i, item in enumerate(self.dataset):
            try:
                paper_id = item['id']
                title = item['title']
                
                # Construct Context from Abstract + Full Text
                abstract = item['abstract']
                full_text_list = []
                
                # QASPER full_text structure handling
                if 'full_text' in item and item['full_text']:
                    ft = item['full_text']
                    if isinstance(ft, dict):
                         # If it's a dict (columnar-like inside the item), it likely has 'section_name' and 'paragraphs' as keys pointing to lists
                         # OR it could be just a dict representing a single section (unlikely if loop failed on keys)
                         # Let's assume it has 'section_name' and 'paragraphs' lists
                         s_names = ft.get('section_name', [])
                         paras = ft.get('paragraphs', [])
                         # Ensure they are lists
                         if isinstance(s_names, list) and isinstance(paras, list):
                             for s_name, p_list in zip(s_names, paras):
                                 if s_name:
                                     full_text_list.append(f"## {s_name}")
                                 if isinstance(p_list, list):
                                     full_text_list.extend(p_list)
                    elif isinstance(ft, list):
                         for section in ft:
                             if isinstance(section, dict):
                                 section_name = section.get('section_name', '')
                                 paragraphs = section.get('paragraphs', [])
                                 if section_name:
                                     full_text_list.append(f"## {section_name}")
                                 full_text_list.extend(paragraphs)
                             elif isinstance(section, str):
                                 # Fallback
                                 full_text_list.append(section)

                
                full_text = "\n\n".join(full_text_list)
                context = f"Title: {title}\n\nAbstract:\n{abstract}\n\nFull Text:\n{full_text}"
                
                # Process Questions
                if 'qas' in item and item['qas']:
                    qas_raw = item['qas']
                    qas_list = []
                    
                    if isinstance(qas_raw, list):
                        qas_list = qas_raw
                    elif isinstance(qas_raw, dict):
                        # Construct list of dicts from columnar dict
                        # Keys: question, question_id, answers, nlp_background, topic_background
                        # We need question, question_id, answers
                        qs = qas_raw.get('question', [])
                        qids = qas_raw.get('question_id', [])
                        ans_lists = qas_raw.get('answers', [])
                        
                        # Ensure equal lengths
                        length = len(qs)
                        if len(qids) == length and len(ans_lists) == length:
                            for q, qid, ans in zip(qs, qids, ans_lists):
                                qas_list.append({
                                    'question': q,
                                    'question_id': qid,
                                    'answers': ans
                                })
                        else:
                            print(f"DEBUG: Mismatched lengths in qas dict for item {i}")

                    for qa in qas_list:
                        if not isinstance(qa, dict):
                            continue
                            
                        question_id = qa.get('question_id', '')
                        question_text = qa.get('question', '')
                        
                        # Get answers
                        answers = []
                        raw_answers = qa.get('answers', [])
                        
                        # raw_answers might be a list of dicts (each having 'answer' key which is a dict)
                        # OR it might be a dict (columnar) containing lists
                        ans_wrappers = []
                        if isinstance(raw_answers, list):
                            ans_wrappers = raw_answers
                        elif isinstance(raw_answers, dict):
                            # Assume columnar: keys are fields of the answer object
                            # We expect 'answer' to be a list of actual answer dicts
                            # But wait, QASPER structure is: answers -> [ { answer: { ... } } ]
                            # So if columnar, raw_answers might be {'answer': [ ... ]}
                            if 'answer' in raw_answers:
                                # This is likely a list of Structs
                                ans_structs = raw_answers['answer']
                                if isinstance(ans_structs, list):
                                     # Reconstruct objects to match expected loop format: {'answer': struct}
                                     # Actually, let's just extract the inner answer structs directly
                                     ans_wrappers = [{'answer': s} for s in ans_structs]
                            else:
                                print(f"DEBUG: raw_answers dict missing 'answer' key in item {i}")
                                
                        for ans_wrapper in ans_wrappers:
                            if isinstance(ans_wrapper, dict):
                                ans_data = ans_wrapper.get('answer', {})
                                # Check different answer types per QASPER schema
                                if ans_data.get('unanswerable', False):
                                    answers.append("Unanswerable")
                                elif ans_data.get('free_form_answer'):
                                    answers.append(ans_data['free_form_answer'])
                                elif ans_data.get('extractive_spans'):
                                    answers.extend(ans_data['extractive_spans'])
                                elif ans_data.get('yes_no') is not None:
                                     answers.append("Yes" if ans_data['yes_no'] else "No")
                        
                        # Combine multiple answers if present (rare but possible)
                        ground_truth = " | ".join(answers) if answers else "Unanswerable"
                        
                        # evidence might be missing or list
                        evidence = []
                        if ans_wrappers:
                             first_ans = ans_wrappers[0].get('answer', {})
                             evidence = first_ans.get('evidence', [])


                        examples.append(QasperExample(
                            id=question_id,
                            title=title,
                            context=context,
                            question=question_text,
                            ground_truth=ground_truth,
                            metadata={"paper_id": paper_id, "evidence": evidence}
                        ))

            except Exception as e:
                print(f"Error processing item {i} (ID: {item.get('id', 'unknown')}): {e}")
                # Print local vars to debug
                import traceback
                traceback.print_exc()
                continue

                    
        return examples

def create_prompts_for_example(example: QasperExample) -> str:
    """Creates a prompt string from a QasperExample."""
    prompt = f"""<context>
{example.context}
</context>

Question: {example.question}

Answer concisely:"""
    return prompt
