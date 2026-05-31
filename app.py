"""
╔══════════════════════════════════════════════════════════════════╗
║  app.py  —  AXON AI Backend Server                              ║
║                                                                  ║
║  DATA FLOW:                                                      ║
║  main.js ──POST /chat──► NLP Engine ──► intents.json            ║
║                                    └──► collected_dataset.txt   ║
║  main.js ──POST /save_log──► collected_dataset.txt              ║
║                                                                  ║
║  Routes:                                                         ║
║    GET  /           → Serve index.html                          ║
║    POST /chat       → Run NLP, return best intent response      ║
║    POST /save_log   → Append conversation to dataset file       ║
║    GET  /health     → Server status check                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import random
import datetime
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ──────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH     = os.path.join(BASE_DIR, "intents.json")
DATASET_PATH     = os.path.join(BASE_DIR, "collected_dataset.txt")
STATIC_DIR       = BASE_DIR          # index.html lives here
CONFIDENCE_THRESHOLD = 0.40          # scores below this → fallback


# ──────────────────────────────────────────────────────────────────
#  FLASK APP + CORS
#  CORS allows main.js to call this server even when the HTML is
#  opened directly as a file (file:// origin) vs localhost origin.
# ──────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=BASE_DIR, template_folder=BASE_DIR)

# Allow all origins (fine for local-only development tool)
# To restrict: CORS(app, origins=["http://127.0.0.1:5000"])
CORS(app)


# ──────────────────────────────────────────────────────────────────
#  NLP ENGINE  (self-contained inside app.py for compact structure)
# ──────────────────────────────────────────────────────────────────

class NLPEngine:
    """
    Lightweight Bag-of-Words NLP classifier.

    Pipeline:
      1. tokenize()        — lowercase + strip punctuation → word list
      2. build_vocab()     — collect all unique words from intents.json
      3. vectorize()       — sentence → binary NumPy array (BoW vector)
      4. cosine_sim()      — measure angle between two BoW vectors
      5. jaccard_sim()     — measure word overlap ratio
      6. classify()        — blend both scores, pick best intent
    """

    def __init__(self, intents_path: str):
        print("[AXON NLP] Loading intents...")
        with open(intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.intents: list[dict] = data["intents"]

        # Build vocabulary (sorted for stable indexing)
        self.vocab: list[str] = self._build_vocab()
        print(f"[AXON NLP] Vocabulary: {len(self.vocab)} words")

        # Pre-compute BoW vectors for every pattern (fast inference)
        # Structure: [(tag, pattern_str, bow_vector, token_set), ...]
        self.index: list[tuple] = self._build_index()
        print(f"[AXON NLP] Indexed {len(self.index)} patterns across "
              f"{len(self.intents)} intents")

    # ── Text preprocessing ─────────────────────────────────────────

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Convert raw text → clean word tokens.
        'Hello, how are YOU?!' → ['hello', 'how', 'are', 'you']
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)   # keep only letters + spaces
        return [w for w in text.split() if w]   # split & drop empty strings

    def _build_vocab(self) -> list[str]:
        """Collect every unique word from all non-fallback patterns."""
        words = set()
        for intent in self.intents:
            if intent["tag"] == "fallback":
                continue
            for pattern in intent["patterns"]:
                words.update(self.tokenize(pattern))
        return sorted(words)   # sorted = stable index order

    def _build_index(self) -> list[tuple]:
        """Pre-vectorize all patterns for fast runtime comparison."""
        index = []
        for intent in self.intents:
            if intent["tag"] == "fallback":
                continue
            for pattern in intent["patterns"]:
                vec       = self._vectorize(pattern)
                token_set = set(self.tokenize(pattern))
                index.append((intent["tag"], pattern, vec, token_set))
        return index

    # ── Feature extraction ─────────────────────────────────────────

    def _vectorize(self, text: str) -> np.ndarray:
        """
        Bag-of-Words encoding.
        Returns a float32 array of length = len(self.vocab).
        Each position = 1.0 if that vocab word appears in text, else 0.0.
        """
        tokens = set(self.tokenize(text))
        return np.array(
            [1.0 if w in tokens else 0.0 for w in self.vocab],
            dtype=np.float32
        )

    # ── Similarity metrics ─────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine Similarity = dot(A,B) / (||A|| × ||B||)
        Range: 0.0 (orthogonal) → 1.0 (identical direction)
        Normalisation handles differing sentence lengths gracefully.
        """
        mag_a = np.linalg.norm(a)
        mag_b = np.linalg.norm(b)
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return float(np.dot(a, b) / (mag_a * mag_b))

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        """
        Jaccard Similarity = |A ∩ B| / |A ∪ B|
        Range: 0.0 (no overlap) → 1.0 (identical sets)
        Great for short inputs like 'hi' or 'thanks'.
        """
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    # ── Classification ─────────────────────────────────────────────

    def classify(self, user_text: str) -> tuple[str, float]:
        """
        Main inference method.

        Steps:
          1. Vectorize the user input
          2. Compute blended score against every indexed pattern
             score = 0.65 × cosine + 0.35 × jaccard
          3. Keep the highest-scoring (tag, score) pair
          4. If best score < CONFIDENCE_THRESHOLD → return 'fallback'

        Returns: (intent_tag: str, confidence: float)
        """
        if not user_text.strip():
            return "fallback", 0.0

        input_vec    = self._vectorize(user_text)
        input_tokens = set(self.tokenize(user_text))

        best_tag   = "fallback"
        best_score = 0.0

        for tag, _pattern, pat_vec, pat_tokens in self.index:
            cos = self._cosine(input_vec, pat_vec)
            jac = self._jaccard(input_tokens, pat_tokens)
            score = (0.65 * cos) + (0.35 * jac)

            if score > best_score:
                best_score = score
                best_tag   = tag

        if best_score < CONFIDENCE_THRESHOLD:
            best_tag = "fallback"

        return best_tag, round(best_score, 4)

    def respond(self, user_text: str) -> dict:
        """
        Full pipeline: raw text → intent → random response.

        Returns dict with keys:
          response  (str)   — the reply text
          tag       (str)   — matched intent label
          score     (float) — confidence (0.0–1.0)
        """
        tag, score = self.classify(user_text)

        # Find matching intent and pick a random response
        for intent in self.intents:
            if intent["tag"] == tag:
                reply = random.choice(intent["responses"])
                return {"response": reply, "tag": tag, "score": score}

        # Safety net (should never reach here)
        return {"response": "Something went wrong on my end.", "tag": "error", "score": 0.0}


# ──────────────────────────────────────────────────────────────────
#  BOOT NLP ENGINE (once at startup — reused for every request)
# ──────────────────────────────────────────────────────────────────

print("=" * 58)
print("  AXON AI — Starting up...")
print("=" * 58)
nlp = NLPEngine(INTENTS_PATH)
print("=" * 58 + "\n")


# ──────────────────────────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    Serve index.html from the project root.
    DATA FLOW: Browser → GET / → index.html → loads main.js
    """
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    ──────────
    DATA FLOW:
      main.js  ──JSON {"message": "..."}──►  here
      here     ──nlp.respond()──►  intents.json
      here     ──JSON {"response":..., "tag":..., "score":...}──►  main.js

    Accepts : { "message": "<user text>" }
    Returns : { "response": "<AI reply>", "tag": "<intent>", "score": <float> }
    """
    data = request.get_json(silent=True)

    # ── Input validation ──────────────────────────────────────────
    if not data or "message" not in data:
        return jsonify({
            "error": "Bad request. Send JSON: {\"message\": \"your text\"}",
            "example": {"message": "Hello!"}
        }), 400

    user_text = str(data["message"]).strip()

    if not user_text:
        return jsonify({
            "response": "Looks like you sent an empty message — try typing something!",
            "tag": "fallback",
            "score": 0.0
        })

    # ── Run NLP inference ─────────────────────────────────────────
    result = nlp.respond(user_text)

    # ── Server-side logging ───────────────────────────────────────
    print(f"[CHAT]  User  : {user_text}")
    print(f"        AXON  : {result['response']}")
    print(f"        Tag   : {result['tag']}  |  Score: {result['score']}\n")

    return jsonify(result)


@app.route("/save_log", methods=["POST"])
def save_log():
    """
    POST /save_log
    ──────────────
    DATA FLOW:
      main.js  ──JSON {"user": "...", "ai": "..."}──►  here
      here     ──append──►  collected_dataset.txt

    Called by main.js immediately after displaying the AI response.
    Builds a local training dataset for future model improvement.

    File format in collected_dataset.txt:
      [2025-01-15 14:32:01] USER: hello
      [2025-01-15 14:32:01]   AI: Hi! I'm AXON...
      ──────────────────────────────────────────
    """
    data = request.get_json(silent=True)

    if not data or "user" not in data or "ai" not in data:
        return jsonify({
            "error": "Bad request. Send JSON: {\"user\": \"...\", \"ai\": \"...\"}"
        }), 400

    user_msg = str(data["user"]).strip()
    ai_msg   = str(data["ai"]).strip()

    if not user_msg or not ai_msg:
        return jsonify({"status": "skipped", "reason": "empty fields"}), 200

    # ── Append to dataset file ────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "─" * 58

    log_entry = (
        f"[{timestamp}] USER: {user_msg}\n"
        f"[{timestamp}]   AI: {ai_msg}\n"
        f"{separator}\n"
    )

    try:
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)

        print(f"[LOG]  Saved conversation to collected_dataset.txt")

        return jsonify({
            "status": "saved",
            "file": "collected_dataset.txt",
            "timestamp": timestamp
        })

    except IOError as e:
        print(f"[LOG ERROR] Could not write to dataset file: {e}")
        return jsonify({"error": f"File write failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple health-check. Returns 200 OK if server is alive."""
    return jsonify({
        "status": "ok",
        "engine": "AXON NLP v2",
        "vocab_size": len(nlp.vocab),
        "patterns_indexed": len(nlp.index),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "dataset_file": DATASET_PATH
    })


# ──────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"  AXON Chatbot running at: http://127.0.0.1:5000")
    print(f"  Dataset logs → {DATASET_PATH}")
    print(f"  Press Ctrl+C to stop.\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True       # hot-reload on code changes during development
    )
