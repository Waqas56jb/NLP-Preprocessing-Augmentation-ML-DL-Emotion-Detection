import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

from src.preprocess import apply_pipeline
from src.suggest import load_and_analyze
from src.model import train_and_eval
from src.augment import synonym_replacement, random_insertion, random_swap, random_deletion, truncate, noise_injection, sentence_shuffle, back_translation, paraphrase
from src.suggest import class_imbalance_suggestion


DEFAULT_DATASET = "test.csv"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion NLP Prototype")
        self.geometry("1000x700")

        self.dataset_path_var = tk.StringVar(value=DEFAULT_DATASET)
        self.text_col_var = tk.StringVar(value="text")
        self.label_col_var = tk.StringVar(value="Emotion")

        self.chk_lower = tk.BooleanVar(value=True)
        self.chk_punct = tk.BooleanVar(value=False)
        self.chk_numbers = tk.BooleanVar(value=False)
        self.chk_ws = tk.BooleanVar(value=True)
        self.chk_stop = tk.BooleanVar(value=False)

        self.model_var = tk.StringVar(value="logreg")

        self.df: pd.DataFrame | None = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Dataset").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.dataset_path_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Browse", command=self._browse).pack(side=tk.LEFT)
        ttk.Button(top, text="Load & Suggest", command=self._load_and_suggest).pack(side=tk.LEFT, padx=5)

        cols = ttk.Frame(self)
        cols.pack(fill=tk.X, padx=10)
        ttk.Label(cols, text="Text column").pack(side=tk.LEFT)
        ttk.Entry(cols, textvariable=self.text_col_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(cols, text="Label column").pack(side=tk.LEFT)
        ttk.Entry(cols, textvariable=self.label_col_var, width=20).pack(side=tk.LEFT, padx=5)

        opts = ttk.LabelFrame(self, text="Preprocessing")
        opts.pack(fill=tk.X, padx=10, pady=10)
        ttk.Checkbutton(opts, text="Lowercase", variable=self.chk_lower).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(opts, text="Remove punctuation", variable=self.chk_punct).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(opts, text="Remove numbers", variable=self.chk_numbers).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(opts, text="Extra whitespace", variable=self.chk_ws).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(opts, text="Remove stopwords", variable=self.chk_stop).pack(side=tk.LEFT, padx=5)

        act = ttk.Frame(self)
        act.pack(fill=tk.X, padx=10)
        ttk.Button(act, text="Preview First 5 (Processed)", command=self._preview_processed).pack(side=tk.LEFT)
        ttk.Button(act, text="Save Processed CSV", command=self._save_processed).pack(side=tk.LEFT, padx=5)

        model_frame = ttk.LabelFrame(self, text="Model")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        for name, val in [
            ("LogReg", "logreg"), ("Naive Bayes", "nb"), ("Linear SVM", "svm"),
            ("RandomForest", "rf"), ("KNN", "knn"), ("XGBoost", "xgb"), ("LightGBM", "lgbm"),
            ("LSTM", "lstm"), ("GRU", "gru"), ("CNN", "cnn"), ("DistilBERT", "distilbert")
        ]:
            ttk.Radiobutton(model_frame, text=name, variable=self.model_var, value=val).pack(side=tk.LEFT)
        ttk.Button(model_frame, text="Train & Evaluate", command=self._train_eval_async).pack(side=tk.LEFT, padx=10)

        aug = ttk.LabelFrame(self, text="Augmentation")
        aug.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(aug, text="Synonym Replace", command=lambda: self._augment_preview("syn")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Random Insert", command=lambda: self._augment_preview("ins")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Random Swap", command=lambda: self._augment_preview("swap")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Random Delete", command=lambda: self._augment_preview("del")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Truncate", command=lambda: self._augment_preview("trunc")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Noise", command=lambda: self._augment_preview("noise")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Sentence Shuffle", command=lambda: self._augment_preview("shuffle")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Back Translation", command=lambda: self._augment_preview("bt")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Paraphrase", command=lambda: self._augment_preview("para")).pack(side=tk.LEFT)
        ttk.Button(aug, text="Save Augmented CSV", command=self._save_augmented).pack(side=tk.LEFT, padx=10)

        self.output = tk.Text(self, height=20)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.dataset_path_var.set(path)

    def _load_and_suggest(self):
        try:
            df, stats, sug = load_and_analyze(self.dataset_path_var.get(), self.text_col_var.get())
            self.df = df
            self.chk_lower.set(sug.get("lowercase", True))
            self.chk_punct.set(sug.get("remove_punctuation", False))
            self.chk_numbers.set(sug.get("remove_numbers", False))
            self.chk_ws.set(sug.get("remove_extra_whitespace", True))
            self.chk_stop.set(sug.get("remove_stopwords", False))

            imb = class_imbalance_suggestion(df, self.label_col_var.get())
            imb_msg = "; class imbalance detected -> consider augmentation" if imb else ""
            self._log(f"Loaded {len(df)} rows{imb_msg}.\n" +
                      f"Punct ratio={stats.punctuation_ratio:.3f}, Digits={stats.digit_ratio:.3f}, WS run={stats.avg_whitespace_run:.2f}, Stopwords={stats.stopword_share:.2f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _current_steps(self):
        steps = []
        if self.chk_lower.get():
            steps.append("lowercase")
        if self.chk_punct.get():
            steps.append("remove_punctuation")
        if self.chk_numbers.get():
            steps.append("remove_numbers")
        if self.chk_ws.get():
            steps.append("remove_extra_whitespace")
        return steps

    def _process_df(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Load a dataset first")
        steps = self._current_steps()
        remove_stop = self.chk_stop.get()
        df = self.df.copy()
        df[self.text_col_var.get()] = df[self.text_col_var.get()].astype(str).apply(lambda t: apply_pipeline(t, steps, remove_stop))
        return df

    def _preview_processed(self):
        try:
            pdf = self._process_df().head(5)
            self._log(pdf.to_string(index=False))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_processed(self):
        try:
            pdf = self._process_df()
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                pdf.to_csv(path, index=False)
                self._log(f"Saved processed dataset to {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _train_eval_async(self):
        def run():
            try:
                dfp = self._process_df()
                res = train_and_eval(dfp, text_col=self.text_col_var.get(), label_col=self.label_col_var.get(), model=self.model_var.get())
                self._log(f"Accuracy: {res.accuracy:.4f}\n\n{res.report}")
            except Exception as e:
                self._log(f"Error: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _augment_preview(self, kind: str):
        try:
            if self.df is None:
                raise RuntimeError("Load a dataset first")
            df = self._process_df()
            col = self.text_col_var.get()
            sample = df[col].astype(str).head(5).tolist()
            out: list[str] = []
            for s in sample:
                if kind == "syn":
                    out.append(synonym_replacement(s, n=1))
                elif kind == "ins":
                    out.append(random_insertion(s, n=1))
                elif kind == "swap":
                    out.append(random_swap(s, n=1))
                elif kind == "del":
                    out.append(random_deletion(s, p=0.2))
                elif kind == "trunc":
                    out.append(truncate(s, max_tokens=20))
                elif kind == "noise":
                    out.append(noise_injection(s, n=1))
                elif kind == "shuffle":
                    out.append(sentence_shuffle(s))
                elif kind == "bt":
                    out.append(back_translation(s))
                elif kind == "para":
                    out.append(paraphrase(s))
                else:
                    out.append(s)
            preview = "\n".join(f"- {o}" for o in out)
            self._log(f"Augmented preview ({kind}):\n{preview}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_augmented(self):
        try:
            if self.df is None:
                raise RuntimeError("Load a dataset first")
            df = self._process_df()
            col = self.text_col_var.get()
            # Simple strategy: for each row, create one augmented variant (synonym replacement)
            aug_texts = df[col].astype(str).apply(lambda s: synonym_replacement(s, n=1))
            aug_df = df.copy()
            aug_df[col] = aug_texts
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                aug_df.to_csv(path, index=False)
                self._log(f"Saved augmented dataset to {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _log(self, msg: str):
        self.output.insert(tk.END, str(msg) + "\n\n")
        self.output.see(tk.END)


if __name__ == "__main__":
    app = App()
    app.mainloop()

