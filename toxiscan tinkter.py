"""Optional desktop interface powered by ToxiScan's shared services."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

from toxiscan_core.classifier import BlockAnalysis, ClassificationError, analyse_blocks
from toxiscan_core.config import DOCUMENT_LIMITS
from toxiscan_core.documents import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    DocumentExtractionError,
    extract_document,
)
from toxiscan_core.text import ExtractedText, prepare_text
from toxiscan_core.web import WebExtractionError, scrape_url

logger = logging.getLogger(__name__)


class ToxiScanDesktop(tk.Tk):
    """Small desktop client that delegates all analysis to the shared core."""

    def __init__(self) -> None:
        super().__init__()
        self.title("🛡️ ToxiScan")
        self.geometry("920x780")
        self.configure(bg="#f0f0f0")
        self.threshold = tk.DoubleVar(value=0.7)
        self.status = tk.StringVar(value="Ready")
        self._build_layout()

    def _build_layout(self) -> None:
        tk.Label(
            self,
            text="Paste text",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
        ).pack(pady=(12, 4))
        self.text_input = scrolledtext.ScrolledText(self, width=105, height=6)
        self.text_input.pack(padx=12)
        tk.Button(
            self,
            text="Analyse text",
            command=self.analyse_pasted_text,
            bg="#0078D7",
            fg="white",
        ).pack(pady=6)

        url_frame = tk.Frame(self, bg="#f0f0f0")
        url_frame.pack(fill="x", padx=12, pady=6)
        tk.Label(url_frame, text="Public page URL", bg="#f0f0f0").pack(side="left")
        self.url_input = tk.Entry(url_frame)
        self.url_input.pack(side="left", fill="x", expand=True, padx=8)
        tk.Button(url_frame, text="Scan page", command=self.analyse_url).pack(side="right")

        controls = tk.Frame(self, bg="#f0f0f0")
        controls.pack(fill="x", padx=12, pady=6)
        tk.Button(
            controls,
            text="Analyse document",
            command=self.analyse_document,
            bg="#28a745",
            fg="white",
        ).pack(side="left")
        tk.Label(controls, text="Detection threshold", bg="#f0f0f0").pack(
            side="left", padx=(24, 4)
        )
        tk.Scale(
            controls,
            variable=self.threshold,
            from_=0.30,
            to=0.90,
            resolution=0.05,
            orient="horizontal",
            bg="#f0f0f0",
        ).pack(side="left")
        tk.Label(controls, textvariable=self.status, bg="#f0f0f0").pack(side="right")

        self.results = scrolledtext.ScrolledText(self, width=105, height=28)
        self.results.pack(fill="both", expand=True, padx=12, pady=(6, 12))

    def _run_in_background(
        self,
        operation: Callable[[], tuple[BlockAnalysis, ...]],
    ) -> None:
        self.status.set("Working…")
        self.results.delete("1.0", tk.END)

        def worker() -> None:
            try:
                result = operation()
            except (DocumentExtractionError, WebExtractionError, ClassificationError) as error:
                message = str(error)
                self.after(0, lambda: self._show_error(message))
            except Exception:
                logger.exception("Unexpected desktop application failure")
                self.after(
                    0,
                    lambda: self._show_error(
                        "ToxiScan could not complete the analysis. Please try again later."
                    ),
                )
            else:
                self.after(0, lambda: self._show_results(result))

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _analyse_extracted(
        extracted: ExtractedText, threshold: float
    ) -> tuple[BlockAnalysis, ...]:
        return analyse_blocks(extracted.blocks, threshold)

    def analyse_pasted_text(self) -> None:
        value = self.text_input.get("1.0", tk.END)
        extracted = prepare_text(value, DOCUMENT_LIMITS.text)
        if not extracted.blocks:
            messagebox.showwarning("Input required", "Enter some readable text to analyse.")
            return
        threshold = self.threshold.get()
        self._run_in_background(lambda: self._analyse_extracted(extracted, threshold))

    def analyse_url(self) -> None:
        url = self.url_input.get().strip()
        if not url:
            messagebox.showwarning("Input required", "Enter a public page URL.")
            return
        threshold = self.threshold.get()

        def operation() -> tuple[BlockAnalysis, ...]:
            return self._analyse_extracted(scrape_url(url).text, threshold)

        self._run_in_background(operation)

    def analyse_document(self) -> None:
        patterns = " ".join(
            f"*{extension}" for extension in sorted(SUPPORTED_DOCUMENT_EXTENSIONS)
        )
        selected = filedialog.askopenfilename(
            filetypes=(("Supported documents", patterns), ("All files", "*.*"))
        )
        if not selected:
            return

        path = Path(selected)
        try:
            document_size = path.stat().st_size
        except OSError:
            self._show_error("The selected document could not be accessed.")
            return
        if document_size > DOCUMENT_LIMITS.max_upload_bytes:
            self._show_error("The selected document exceeds the processing limit.")
            return
        threshold = self.threshold.get()

        def operation() -> tuple[BlockAnalysis, ...]:
            extracted = extract_document(path.name, path.read_bytes())
            return self._analyse_extracted(extracted, threshold)

        self._run_in_background(operation)

    def _show_error(self, message: str) -> None:
        self.status.set("Failed")
        messagebox.showerror("Analysis failed", message)

    def _show_results(self, analyses: tuple[BlockAnalysis, ...]) -> None:
        flagged = [analysis for analysis in analyses if analysis.flagged]
        if not flagged:
            self.results.insert(
                tk.END, "No potentially harmful language was detected.\n"
            )
        for index, analysis in enumerate(flagged, start=1):
            labels = ", ".join(category.label for category in analysis.categories)
            self.results.insert(tk.END, f"{index}. {labels}\n{analysis.text}\n")
            for evidence in analysis.evidence:
                self.results.insert(tk.END, f"   Evidence: {evidence.text}\n")
            self.results.insert(tk.END, "\n")
        self.status.set(f"Complete — {len(flagged)} of {len(analyses)} sections flagged")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    ToxiScanDesktop().mainloop()


if __name__ == "__main__":
    main()
