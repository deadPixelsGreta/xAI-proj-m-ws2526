"""Rich console UI utilities for human-readable training output.

This module provides beautiful, colorful, and informative console output
for Google Colab and terminal environments.
"""

import time
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Color Codes
# ═══════════════════════════════════════════════════════════════════════════════


class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


# ═══════════════════════════════════════════════════════════════════════════════
# Box Drawing Characters
# ═══════════════════════════════════════════════════════════════════════════════


class Box:
    """Unicode box-drawing characters."""

    # Single line
    H = "─"  # Horizontal
    V = "│"  # Vertical
    TL = "┌"  # Top-left
    TR = "┐"  # Top-right
    BL = "└"  # Bottom-left
    BR = "┘"  # Bottom-right
    LT = "├"  # Left-T
    RT = "┤"  # Right-T
    TT = "┬"  # Top-T
    BT = "┴"  # Bottom-T
    X = "┼"  # Cross

    # Double line
    DH = "═"  # Double horizontal
    DV = "║"  # Double vertical
    DTL = "╔"  # Double top-left
    DTR = "╗"  # Double top-right
    DBL = "╚"  # Double bottom-left
    DBR = "╝"  # Double bottom-right

    # Mixed
    MTL = "╒"  # Mixed top-left (double H, single V)
    MTR = "╕"
    MBL = "╘"
    MBR = "╛"


# ═══════════════════════════════════════════════════════════════════════════════
# Symbols (ASCII-only for Colab compatibility)
# ═══════════════════════════════════════════════════════════════════════════════


class Icons:
    """ASCII symbols for status indicators (Colab-compatible)."""

    CHECK = "[OK]"
    CROSS = "[X]"
    ARROW_RIGHT = ">"
    ARROW_DOWN = "v"
    STAR = "*"
    BULLET = "-"
    ROCKET = ">>>"
    CHART = "[#]"
    BRAIN = "[~]"
    GEAR = "[*]"
    PACKAGE = "[=]"
    CLOCK = "[T]"
    SPARKLES = "***"
    FIRE = "(!)"
    TARGET = "[>]"
    TROPHY = "[!]"
    WARNING = "[!]"
    SUCCESS = "[OK]"
    STOP = "[X]"
    SAVE = "[S]"
    GPU = "[G]"
    LAYERS = "[L]"
    CHART_UP = "[^]"
    CHART_DOWN = "[v]"


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting Functions
# ═══════════════════════════════════════════════════════════════════════════════


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def format_params(params: int) -> str:
    """Format parameter count with magnitude suffix."""
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K"
    return str(params)


def color(text: str, *styles) -> str:
    """Apply ANSI color/style codes to text."""
    codes = "".join(styles)
    return f"{codes}{text}{Colors.RESET}"


def center_text(text: str, width: int, fill_char: str = " ") -> str:
    """Center text within a given width."""
    text_len = len(text)
    if text_len >= width:
        return text
    left_pad = (width - text_len) // 2
    right_pad = width - text_len - left_pad
    return fill_char * left_pad + text + fill_char * right_pad


# ═══════════════════════════════════════════════════════════════════════════════
# Console UI Components
# ═══════════════════════════════════════════════════════════════════════════════


class ConsoleUI:
    """Rich console UI for training output."""

    WIDTH = 70  # Default display width

    @classmethod
    def header(cls, title: str, subtitle: str = None):
        """Print a beautiful header box."""
        width = cls.WIDTH

        print()
        print(color(f"{Box.DTL}{Box.DH * (width - 2)}{Box.DTR}", Colors.BRIGHT_CYAN))
        print(
            color(f"{Box.DV}", Colors.BRIGHT_CYAN)
            + color(center_text(title, width - 2), Colors.BOLD, Colors.WHITE)
            + color(f"{Box.DV}", Colors.BRIGHT_CYAN)
        )
        if subtitle:
            print(
                color(f"{Box.DV}", Colors.BRIGHT_CYAN)
                + color(center_text(subtitle, width - 2), Colors.DIM)
                + color(f"{Box.DV}", Colors.BRIGHT_CYAN)
            )
        print(color(f"{Box.DBL}{Box.DH * (width - 2)}{Box.DBR}", Colors.BRIGHT_CYAN))
        print()

    @classmethod
    def section(cls, title: str):
        """Print a section divider."""
        width = cls.WIDTH
        padding = width - len(title) - 4

        print()
        print(
            color(f"{Box.TL}{Box.H * 2}", Colors.CYAN)
            + color(f" {title} ", Colors.BOLD, Colors.CYAN)
            + color(f"{Box.H * padding}{Box.TR}", Colors.CYAN)
        )

    @classmethod
    def subsection(cls, title: str):
        """Print a subsection header."""
        print(f"\n  {color('-', Colors.CYAN)} {color(title, Colors.BOLD)}")

    @classmethod
    def key_value(
        cls,
        key: str,
        value: str,
        indent: int = 4,
        key_color=Colors.BRIGHT_BLACK,
        value_color=Colors.WHITE,
    ):
        """Print a key-value pair."""
        padding = " " * indent
        print(
            f"{padding}{color(key + ':', key_color)} {color(str(value), value_color)}"
        )

    @classmethod
    def info(cls, message: str):
        """Print an info message."""
        print(f"    {color('>', Colors.CYAN)} {message}")

    @classmethod
    def success(cls, message: str):
        """Print a success message."""
        print(f"    {color('[OK]', Colors.GREEN)} {color(message, Colors.GREEN)}")

    @classmethod
    def warning(cls, message: str):
        """Print a warning message."""
        print(f"    {color('[!]', Colors.YELLOW)} {color(message, Colors.YELLOW)}")

    @classmethod
    def error(cls, message: str):
        """Print an error message."""
        print(f"    {color('[X]', Colors.RED)} {color(message, Colors.RED)}")

    @classmethod
    def divider(cls, style: str = "light"):
        """Print a divider line."""
        if style == "light":
            print(color(f"    {Box.H * (cls.WIDTH - 8)}", Colors.BRIGHT_BLACK))
        elif style == "heavy":
            print(color(f"  {Box.DH * (cls.WIDTH - 4)}", Colors.BRIGHT_BLACK))

    @classmethod
    def table_row(cls, columns: List[Tuple[str, int]], colors_list: List = None):
        """Print a formatted table row."""
        if colors_list is None:
            colors_list = [Colors.WHITE] * len(columns)

        row = "    "
        for i, (text, width) in enumerate(columns):
            row += color(f"{text:<{width}}", colors_list[i])
        print(row)

    @classmethod
    def progress_bar(
        cls,
        current: int,
        total: int,
        width: int = 30,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Create a text-based progress bar."""
        filled = int(width * current / total)
        empty = width - filled
        bar = color("█" * filled, Colors.GREEN) + color(
            "░" * empty, Colors.BRIGHT_BLACK
        )
        percent = current / total * 100
        return f"{prefix} [{bar}] {percent:5.1f}% {suffix}"

    @classmethod
    def stats_box(cls, title: str, stats: Dict[str, str]):
        """Print a statistics box."""
        print()
        print(f"  {color(title, Colors.BOLD, Colors.CYAN)}")
        print(
            color(f"  {Box.TL}{Box.H * (cls.WIDTH - 6)}{Box.TR}", Colors.BRIGHT_BLACK)
        )

        for key, value in stats.items():
            key_display = f"  {Box.V}   {key}:"
            padding = cls.WIDTH - len(key_display) - len(str(value)) - 4
            print(
                color(f"  {Box.V}", Colors.BRIGHT_BLACK)
                + f"   {color(key, Colors.WHITE)}: "
                + " " * padding
                + color(str(value), Colors.BRIGHT_CYAN, Colors.BOLD)
                + color(f"   {Box.V}", Colors.BRIGHT_BLACK)
            )

        print(
            color(f"  {Box.BL}{Box.H * (cls.WIDTH - 6)}{Box.BR}", Colors.BRIGHT_BLACK)
        )

    @classmethod
    def backbone_loaded(cls, name: str, features: int, index: int, total: int):
        """Print backbone loading status."""
        status = color(f"[{index}/{total}]", Colors.BRIGHT_BLACK)
        name_str = color(name, Colors.BOLD, Colors.WHITE)
        features_str = color(f"{features:,} features", Colors.CYAN)
        print(
            f"    {status} {color('[OK]', Colors.GREEN)} Loaded {name_str} ({features_str})"
        )

    @classmethod
    def epoch_header(cls, epoch: int, total_epochs: int, lr: float = None):
        """Print epoch header."""
        progress = f"[{epoch}/{total_epochs}]"
        lr_str = f" │ LR: {lr:.2e}" if lr else ""

        print()
        print(
            color(f"  {Box.TL}{Box.H * 3}", Colors.MAGENTA)
            + color(f" EPOCH {progress} ", Colors.BOLD, Colors.MAGENTA)
            + color(
                f"{Box.H * (cls.WIDTH - 20 - len(progress))}{Box.TR}", Colors.MAGENTA
            )
        )
        if lr_str:
            print(
                color(f"  {Box.V}", Colors.MAGENTA)
                + f"  Learning Rate: {color(f'{lr:.2e}', Colors.YELLOW)}"
            )

    @classmethod
    def epoch_summary(
        cls,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        time_elapsed: float,
        is_best: bool = False,
    ):
        """Print epoch summary with training and validation metrics."""
        print()
        print(f"  {Box.V}")

        # Training metrics
        print(
            f"  {Box.LT}{Box.H * 2} {color('Training', Colors.BOLD)} "
            + color(f"{Box.H * (cls.WIDTH - 16)}{Box.RT}", Colors.BRIGHT_BLACK)
        )
        print(
            f"  {Box.V}   Loss: {color(f'{train_loss:.4f}', Colors.YELLOW):<20} "
            + f"Accuracy: {color(f'{train_acc:.2f}%', Colors.GREEN if train_acc > 50 else Colors.YELLOW)}"
        )

        # Validation metrics
        print(
            f"  {Box.LT}{Box.H * 2} {color('Validation', Colors.BOLD)} "
            + color(f"{Box.H * (cls.WIDTH - 18)}{Box.RT}", Colors.BRIGHT_BLACK)
        )

        acc_color = Colors.GREEN if val_acc > 50 else Colors.YELLOW
        best_indicator = (
            color(" <-- NEW BEST!", Colors.BRIGHT_GREEN, Colors.BOLD) if is_best else ""
        )

        print(
            f"  {Box.V}   Loss: {color(f'{val_loss:.4f}', Colors.YELLOW):<20} "
            + f"Accuracy: {color(f'{val_acc:.2f}%', acc_color)}{best_indicator}"
        )

        # Time
        print(f"  {Box.V}")
        print(
            f"  {Box.V}   Epoch Time: {color(format_time(time_elapsed), Colors.CYAN)}"
        )
        print(f"  {Box.BL}{Box.H * (cls.WIDTH - 4)}{Box.BR}")

    @classmethod
    def training_complete(
        cls,
        best_acc: float,
        total_time: float,
        total_epochs: int,
        save_path: str = None,
    ):
        """Print training completion summary."""
        print()
        print(color(f"{Box.DTL}{Box.DH * (cls.WIDTH - 2)}{Box.DTR}", Colors.GREEN))
        print(
            color(f"{Box.DV}", Colors.GREEN)
            + color(
                center_text("TRAINING COMPLETE", cls.WIDTH - 2),
                Colors.BOLD,
                Colors.WHITE,
            )
            + color(f"{Box.DV}", Colors.GREEN)
        )
        print(color(f"{Box.DBL}{Box.DH * (cls.WIDTH - 2)}{Box.DBR}", Colors.GREEN))

        print(
            f"\n  {color('Best Validation Accuracy:', Colors.BOLD)} "
            + color(f"{best_acc:.2f}%", Colors.GREEN, Colors.BOLD)
        )
        print(
            f"  {color('Total Training Time:', Colors.BOLD)} "
            + color(format_time(total_time), Colors.CYAN)
        )
        print(
            f"  {color('Epochs Completed:', Colors.BOLD)} "
            + color(str(total_epochs), Colors.CYAN)
        )

        if save_path:
            print(f"\n  {color('Model saved to:', Colors.BOLD)}")
            print(f"     {color(save_path, Colors.BRIGHT_CYAN, Colors.UNDERLINE)}")

        print()

    @classmethod
    def early_stopping(cls, patience: int, best_acc: float):
        """Print early stopping message."""
        print()
        print(color(f"  {Box.TL}{Box.H * (cls.WIDTH - 4)}{Box.TR}", Colors.YELLOW))
        print(
            color(f"  {Box.V}", Colors.YELLOW)
            + center_text("EARLY STOPPING TRIGGERED", cls.WIDTH - 4)
            + color(f"{Box.V}", Colors.YELLOW)
        )
        print(color(f"  {Box.BL}{Box.H * (cls.WIDTH - 4)}{Box.BR}", Colors.YELLOW))
        print(
            f"\n  [!] No improvement for {color(str(patience), Colors.YELLOW)} epochs"
        )
        print(f"  Best validation accuracy: {color(f'{best_acc:.2f}%', Colors.GREEN)}")
        print()


class TrainingTimer:
    """Timer utility for tracking training duration."""

    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        self.epoch_times = []

    def start(self):
        """Start the training timer."""
        self.start_time = time.time()

    def start_epoch(self):
        """Start timing an epoch."""
        self.epoch_start_time = time.time()

    def end_epoch(self) -> float:
        """End timing an epoch and return elapsed time."""
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        return elapsed

    def total_elapsed(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time

    def avg_epoch_time(self) -> float:
        """Get average epoch time."""
        if not self.epoch_times:
            return 0
        return sum(self.epoch_times) / len(self.epoch_times)

    def eta(self, current_epoch: int, total_epochs: int) -> str:
        """Estimate time remaining."""
        if not self.epoch_times:
            return "Calculating..."
        avg_time = self.avg_epoch_time()
        remaining = (total_epochs - current_epoch) * avg_time
        return format_time(remaining)
