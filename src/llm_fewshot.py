"""Few-shot *prompting* with a local LLM through Ollama (works on a Mac, no GPU code).

  brew install ollama && ollama pull llama3.1:8b   (or a smaller model, e.g. llama3.2:3b)
  python -m src.llm_fewshot "Wah re sarkar, khaad toh aadhi mitti nikli!"
"""
import sys, json, requests, pandas as pd, os

def build_prompt(query, shots):
    ex = "\n".join(f'Query: {t}\nLabel: {"SARCASTIC_GRIEVANCE" if y else "LITERAL"}\n' for t, y in shots)
    return ("You classify farmer queries (English, Hindi, Hinglish). A query is SARCASTIC_GRIEVANCE when "
            "positive wording hides a complaint about a bad situation; otherwise LITERAL. "
            "Answer with only the label.\n\n" + ex + f"Query: {query}\nLabel:")

def classify(query, k=3, model="llama3.1:8b", seed=0):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "agri_sarcasm_seed.csv"))
    sh = pd.concat([df[df.label == c].sample(k, random_state=seed) for c in (0, 1)]).sample(frac=1, random_state=seed)
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": build_prompt(query, list(zip(sh.text, sh.label))),
                            "stream": False, "options": {"temperature": 0}}, timeout=120)
    return r.json()["response"].strip()

if __name__ == "__main__":
    print(classify(" ".join(sys.argv[1:])))


def advise(query, language="English", model="llama3.1:8b"):
    """Short advisory answer in the farmer's own language via local Ollama."""
    prompt = ("You are an agricultural advisor for Indian farmers. Answer the farmer's question "
              f"in at most 4 short sentences, written in {language}. Be practical and safe; "
              "if unsure, say an expert should be consulted.\n\n" + f"Question: {query}\nAnswer:")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.3}}, timeout=180)
    return r.json()["response"].strip()
