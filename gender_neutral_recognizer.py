"""
Gender-Neutral Language Recognizer
==================================
Uses a hand-built Finite State Automaton (FSA) to find gender-specific terms
in .txt, .docx and .pdf files and suggests gender-neutral alternatives.

Install the libraries once:
    pip install python-docx pdfplumber

Run the window (GUI):
    python gender_neutral_recognizer.py

Run on one file from the command line:
    python gender_neutral_recognizer.py myfile.txt

Write the FSA transition table to a CSV file (for your documentation):
    python gender_neutral_recognizer.py --dump-fsa fsa_table.csv
"""

import argparse
import csv
import os
import sys
from collections import Counter, OrderedDict

NEUTRAL_MESSAGE = "The text has gender-neutral language."


# =============================================================================
# 1. TERM LIST  (term -> list of gender-neutral alternatives)
# =============================================================================
def build_terms():
    """Return the predefined list of gender-specific terms and replacements."""
    terms = OrderedDict()

    # --- Pronouns ---------------------------------------------------------
    terms.update({
        "he": ["they"],
        "she": ["they"],
        "him": ["them"],
        "her": ["them", "their"],          # depends on the sentence
        "his": ["their", "theirs"],        # depends on the sentence
        "hers": ["theirs"],
        "himself": ["themselves", "themself"],
        "herself": ["themselves", "themself"],
        # Phrases (the FSA reads the spaces and slashes too)
        "he or she": ["they"],
        "she or he": ["they"],
        "he/she": ["they"],
        "s/he": ["they"],
        "him or her": ["them"],
        "her or him": ["them"],
        "him/her": ["them"],
        "his or her": ["their"],
        "her or his": ["their"],
        "his/her": ["their"],
        "himself or herself": ["themselves"],
        "herself or himself": ["themselves"],
    })

    # --- "-man" words: the -man, -men, -woman and -women forms are made
    #     automatically from this table ---------------------------------------
    man_words = {
        "chairman": ["chairperson", "chair"],
        "policeman": ["police officer"],
        "fireman": ["firefighter"],
        "salesman": ["salesperson", "sales representative"],
        "businessman": ["businessperson"],
        "spokesman": ["spokesperson"],
        "congressman": ["legislator", "representative"],
        "foreman": ["supervisor"],
        "workman": ["worker"],
        "craftsman": ["craftsperson", "artisan"],
        "mailman": ["mail carrier"],
        "postman": ["mail carrier", "postal worker"],
        "cameraman": ["camera operator"],
        "anchorman": ["anchor"],
        "repairman": ["repair technician"],
        "layman": ["layperson"],
        "tradesman": ["tradesperson"],
        "sportsman": ["athlete"],
        "fisherman": ["fisher"],
        "handyman": ["maintenance worker"],
        "deliveryman": ["delivery driver"],
        "middleman": ["intermediary"],
        "freshman": ["first-year student"],
    }
    no_woman_form = {"freshman", "middleman"}
    for man, alts in man_words.items():
        stem = man[:-3]
        plural_alts = [a + "s" for a in alts]
        terms[man] = alts
        terms[stem + "men"] = plural_alts
        if man not in no_woman_form:
            terms[stem + "woman"] = alts
            terms[stem + "women"] = plural_alts

    # --- Other gendered words ---------------------------------------------
    terms.update({
        "stewardess": ["flight attendant"],
        "stewardesses": ["flight attendants"],
        "steward": ["flight attendant"],
        "stewards": ["flight attendants"],
        "waiter": ["server"],
        "waiters": ["servers"],
        "waitress": ["server"],
        "waitresses": ["servers"],
        "actress": ["actor"],
        "actresses": ["actors"],
        "hostess": ["host"],
        "hostesses": ["hosts"],
        "headmaster": ["head teacher", "principal"],
        "headmistress": ["head teacher", "principal"],
        "landlord": ["property owner", "landowner"],
        "landlady": ["property owner", "landowner"],
        "housewife": ["homemaker"],
        "housewives": ["homemakers"],
        "mankind": ["humankind", "humanity"],
        "manpower": ["workforce", "staff"],
        "man-made": ["artificial", "synthetic"],
        "manmade": ["artificial", "synthetic"],
        "man hours": ["work hours", "person-hours"],
        "sportsmanship": ["fair play", "good conduct"],
        "forefathers": ["ancestors", "forebears"],
        "brotherhood": ["community", "solidarity"],
        "guys": ["everyone", "folks"],
        "ladies and gentlemen": ["everyone", "honored guests"],
        "boys and girls": ["children", "everyone"],
    })
    return terms


# =============================================================================
# 2. THE FINITE STATE AUTOMATON
# =============================================================================
class Match:
    """One gender-specific term found in the text."""

    def __init__(self, start, end, found, term, alternatives, line):
        self.start = start            # index of first character
        self.end = end                # index after the last character
        self.found = found            # text as written, e.g. "Chairman"
        self.term = term              # term in the list, e.g. "chairman"
        self.alternatives = alternatives
        self.line = line              # 1-based line number


class FSA:
    """
    A deterministic finite automaton that is built as a trie.

        Q  = states (numbers 0, 1, 2, ...)
        q0 = state 0, the start state
        Σ  = letters, digits, space, '-', '/', ...
        δ  = self.trans[state][character] -> next state
        F  = accepting states, stored in self.accept

    If there is no transition for a character, the machine goes to the
    implicit DEAD state (None) and the scan restarts from q0.
    """

    START = 0

    def __init__(self, terms):
        self.trans = [{}]      # trans[state] = {character: next_state}
        self.accept = {}       # accepting state -> (term, alternatives)
        for term, alternatives in terms.items():
            state = self.START
            for ch in term.lower():
                nxt = self.trans[state].get(ch)
                if nxt is None:
                    self.trans.append({})
                    nxt = len(self.trans) - 1
                    self.trans[state][ch] = nxt
                state = nxt
            self.accept[state] = (term, alternatives)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _symbol(ch):
        """Any white space (space, tab, newline) counts as one space."""
        return " " if ch.isspace() else ch.lower()

    def _longest_match(self, text, i):
        """Run the FSA from q0 at position i.
        Return (end_index, accepting_state) of the longest whole-word match,
        or None."""
        state = self.START
        j = i
        n = len(text)
        best = None
        while j < n:
            state = self.trans[state].get(self._symbol(text[j]))
            if state is None:                 # dead state
                break
            j += 1
            at_word_end = (j == n) or not text[j].isalnum()
            if state in self.accept and at_word_end:
                best = (j, state)             # remember it, keep looking
        return best

    # -- main scan -------------------------------------------------------
    def scan(self, text):
        matches = []
        i, n = 0, len(text)
        while i < n:
            starts_word = text[i].isalpha() and (i == 0 or not text[i - 1].isalnum())
            if starts_word:
                result = self._longest_match(text, i)
                if result:
                    end, state = result
                    term, alts = self.accept[state]
                    line = text.count("\n", 0, i) + 1
                    matches.append(Match(i, end, text[i:end], term, alts, line))
                    i = end
                    continue
            i += 1
        return matches

    # -- for documentation -----------------------------------------------
    def dump_table(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["state", "input", "next_state", "next_is_accepting", "term"])
            for s, moves in enumerate(self.trans):
                for ch, nxt in moves.items():
                    term = self.accept[nxt][0] if nxt in self.accept else ""
                    w.writerow([f"q{s}", "SPACE" if ch == " " else ch, f"q{nxt}",
                                "yes" if nxt in self.accept else "no", term])


# =============================================================================
# 3. FILE READING  (.txt, .docx, .pdf)
# =============================================================================
class FileReadError(Exception):
    pass


def read_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        raw = open(path, "rb").read()
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    elif ext == ".docx":
        try:
            import docx
        except ImportError:
            raise FileReadError("Install python-docx first:  pip install python-docx")
        document = docx.Document(path)
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        text = "\n".join(parts)
    elif ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise FileReadError("Install pdfplumber first:  pip install pdfplumber")
        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        if not text.strip():
            raise FileReadError("No text found in this PDF (it may be a scanned image).")
    elif ext == ".doc":
        raise FileReadError("Old .doc files are not supported. "
                            "Open it in Word and use Save As -> .docx.")
    else:
        raise FileReadError(f"Unsupported file type: {ext or 'unknown'}. "
                            "Use .txt, .docx or .pdf.")

    return text.replace("\r\n", "\n").replace("\r", "\n")


# =============================================================================
# 4. REPORT
# =============================================================================
def make_report(matches, text, source_name="(pasted text)"):
    if not matches:
        return NEUTRAL_MESSAGE

    counts = Counter(m.term for m in matches)
    alts_of = {m.term: m.alternatives for m in matches}
    lines = text.split("\n")

    out = []
    out.append("GENDER-NEUTRAL LANGUAGE REPORT")
    out.append("=" * 60)
    out.append(f"Source: {source_name}")
    out.append(f"Gender-specific terms found: {len(matches)} "
               f"({len(counts)} different)")
    out.append("")
    out.append("SUMMARY")
    out.append("-" * 60)
    width = max(len(t) for t in counts) + 2
    out.append(f"{'Term'.ljust(width)}{'Count'.ljust(8)}Suggested replacement(s)")
    for term, count in counts.most_common():
        out.append(f"{term.ljust(width)}{str(count).ljust(8)}{' / '.join(alts_of[term])}")

    out.append("")
    out.append("DETAILS")
    out.append("-" * 60)
    for m in matches:
        context = lines[m.line - 1].strip()
        if len(context) > 90:
            context = context[:87] + "..."
        out.append(f"Line {m.line}: \"{m.found}\" -> {' / '.join(m.alternatives)}")
        out.append(f"    {context}")
    return "\n".join(out)


def analyze_text(fsa, text, source_name="(pasted text)"):
    matches = fsa.scan(text)
    return matches, make_report(matches, text, source_name)


# =============================================================================
# 5. GRAPHICAL INTERFACE (Tkinter)
# =============================================================================
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    fsa = FSA(build_terms())
    state = {"source": "(pasted text)", "report": ""}

    root = tk.Tk()
    root.title("Gender-Neutral Language Recognizer")
    root.geometry("1100x650")

    # --- top bar ---
    bar = ttk.Frame(root, padding=8)
    bar.pack(fill="x")
    file_label = ttk.Label(bar, text="No file selected")

    def scan():
        text = doc.get("1.0", "end-1c")
        doc.tag_remove("hit", "1.0", "end")
        if not text.strip():
            messagebox.showinfo("Nothing to scan", "Upload a file or paste some text first.")
            return
        matches, report = analyze_text(fsa, text, state["source"])
        for m in matches:
            doc.tag_add("hit", f"1.0+{m.start}c", f"1.0+{m.end}c")
        state["report"] = report
        out.config(state="normal")
        out.delete("1.0", "end")
        out.insert("1.0", report)
        out.config(state="disabled")
        status.set(NEUTRAL_MESSAGE if not matches
                   else f"{len(matches)} gender-specific term(s) found.")

    def upload():
        path = filedialog.askopenfilename(
            title="Choose a document",
            filetypes=[("Documents", "*.txt *.docx *.pdf *.doc"), ("All files", "*.*")])
        if not path:
            return
        try:
            text = read_file(path)
        except FileReadError as e:
            messagebox.showerror("Cannot read file", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"Could not read the file:\n{e}")
            return
        state["source"] = os.path.basename(path)
        file_label.config(text=state["source"])
        doc.delete("1.0", "end")
        doc.insert("1.0", text)
        scan()

    def save_report():
        if not state["report"]:
            messagebox.showinfo("No report", "Scan some text first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text file", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(state["report"])

    ttk.Button(bar, text="Upload File", command=upload).pack(side="left")
    ttk.Button(bar, text="Scan Text", command=scan).pack(side="left", padx=6)
    ttk.Button(bar, text="Save Report", command=save_report).pack(side="left")
    file_label.pack(side="left", padx=12)

    # --- two panes: document (left) and report (right) ---
    panes = ttk.PanedWindow(root, orient="horizontal")
    panes.pack(fill="both", expand=True, padx=8, pady=4)

    left = ttk.LabelFrame(panes, text="Document (you can also paste or type here)")
    doc = scrolledtext.ScrolledText(left, wrap="word", font=("Segoe UI", 10))
    doc.pack(fill="both", expand=True)
    doc.tag_config("hit", background="#ffe066")
    panes.add(left, weight=1)

    right = ttk.LabelFrame(panes, text="Report")
    out = scrolledtext.ScrolledText(right, wrap="word", font=("Consolas", 10),
                                    state="disabled")
    out.pack(fill="both", expand=True)
    panes.add(right, weight=1)

    status = tk.StringVar(value="Upload a .txt, .docx or .pdf file to begin.")
    ttk.Label(root, textvariable=status, relief="sunken", anchor="w",
              padding=4).pack(fill="x", side="bottom")

    root.mainloop()


# =============================================================================
# 6. ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Gender-Neutral Language Recognizer")
    parser.add_argument("file", nargs="?", help="a .txt, .docx or .pdf file to scan")
    parser.add_argument("--dump-fsa", metavar="CSV",
                        help="write the FSA transition table to a CSV file and exit")
    args = parser.parse_args()

    if args.dump_fsa:
        fsa = FSA(build_terms())
        fsa.dump_table(args.dump_fsa)
        print(f"States: {len(fsa.trans)}  Accepting states: {len(fsa.accept)}")
        print(f"Transition table written to {args.dump_fsa}")
        return

    if args.file:
        try:
            text = read_file(args.file)
        except (FileReadError, OSError) as e:
            sys.exit(f"Error: {e}")
        _, report = analyze_text(FSA(build_terms()), text, os.path.basename(args.file))
        print(report)
        return

    run_gui()


if __name__ == "__main__":
    main()
