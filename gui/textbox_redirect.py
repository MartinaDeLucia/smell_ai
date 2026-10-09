import io
import queue
import tkinter as tk


class TextBoxRedirect(io.StringIO):
    """Thread-safe stdout redirect for a Tkinter Text widget."""

    def __init__(self, textbox, poll_ms: int = 50):
        super().__init__()
        self.textbox = textbox
        self.poll_ms = poll_ms
        self._queue: queue.Queue[str] = queue.Queue()
        # Scheduled from the UI thread during construction.
        self.textbox.after(self.poll_ms, self._drain_queue)

    def write(self, text):
        if text:
            self._queue.put(str(text))
        return len(text or "")

    def _drain_queue(self):
        chunks = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if chunks:
            self.textbox.config(state="normal")
            self.textbox.insert(tk.END, "".join(chunks))
            self.textbox.config(state="disabled")
            self.textbox.see(tk.END)

        if self.textbox.winfo_exists():
            self.textbox.after(self.poll_ms, self._drain_queue)

    def flush(self):
        pass
