# trailing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import time

Side = Literal["buy", "sell"]
Mode = Literal["percent", "atr", "swing", "chandelier"]


@dataclass
class TrailConfig:
    mode: Mode = "chandelier"  # percent | atr | swing | chandelier
    pct: float = 0.01  # 1% trail for mode="percent"
    atr_k: float = 3.0  # k for atr/chandelier
    atr_value: Optional[float] = None  # provide latest ATR if using atr/chandelier
    swing_level: Optional[float] = None  # latest confirmed swing low/high (per side)
    be_trigger_atr: float = 1.0  # move to BE after +1 ATR move
    min_improve_atr: float = 0.25  # min improvement step in ATR units
    cooldown_s: float = 5.0  # min seconds between updates (0 = no cooldown)
    spread_buffer: float = 0.0  # add/subtract to reduce noise hits
    price_scale: int = 4  # decimal places for rounding on this symbol


@dataclass
class TrailState:
    armed: bool = False
    last_update_ts: float = 0.0
    stop: Optional[float] = None
    peak: Optional[float] = None  # highest high since entry for long; lowest low for short
    locked: bool = False


class TrailingStopEngine:
    def __init__(self, side: Side, entry: float, cfg: TrailConfig) -> None:
        self.side = side
        self.entry = float(entry)
        self.cfg = cfg
        self.state = TrailState(armed=True, peak=self.entry)

    def _round(self, px: float) -> float:
        return float(f"{px:.{self.cfg.price_scale}f}")

    def _improved_enough(self, new: float, old: Optional[float]) -> bool:
        """Return True if new stop improves meaningfully compared to old."""
        if old is None:
            return True
        if self.cfg.atr_value:
            min_change = self.cfg.min_improve_atr * self.cfg.atr_value
            return abs(new - old) >= min_change
        return new != old

    def _cooldown_ok(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        if self.state.last_update_ts == 0.0:
            return True
        if self.cfg.cooldown_s == 0.0:
            return True
        return (now - self.state.last_update_ts) >= self.cfg.cooldown_s
    

    def _candidate_stop(self, last: float) -> Optional[float]:
        """Compute a new stop level candidate based on current mode and side."""
        mode = self.cfg.mode
        atr = self.cfg.atr_value or 0.0

        if mode == "percent":
            if self.side == "buy":
                return last * (1 - self.cfg.pct)
            else:
                return last * (1 + self.cfg.pct)

        elif mode == "atr":
            if atr == 0:
                return None
            if self.side == "buy":
                return last - (self.cfg.atr_k * atr)
            else:
                return last + (self.cfg.atr_k * atr)

        elif mode == "chandelier":
            # Always derive from the most recent peak, even on retraces
            if atr == 0 or self.state.peak is None:
                return None
            if self.side == "buy":
                return self.state.peak - (self.cfg.atr_k * atr)
            else:
                return self.state.peak + (self.cfg.atr_k * atr)

        elif mode == "swing":
            # Use provided swing level directly
            return self.cfg.swing_level

        return None
    

    def _breakeven_ready(self, last: float) -> bool:
        if self.cfg.atr_value is None:
            return False
        advance = (last - self.entry) if self.side == "buy" else (self.entry - last)
        return advance >= (self.cfg.be_trigger_atr * self.cfg.atr_value)
    
    def on_price(self, last: Optional[float]) -> Optional[float]:
        """Handle incoming price and update trailing stop."""
        if last is None:
            return None

        last = float(abs(last) or 1e-8)
        now = time.time()

        # Respect lock
        if self.state.locked:
            return self.state.stop

        # === Update peak ===
        if self.state.peak is None:
            self.state.peak = last
        elif self.side == "buy":
            self.state.peak = max(self.state.peak, last)
        else:
            self.state.peak = min(self.state.peak, last)

        # === Chandelier mode special: always compute from peak ===
        if self.cfg.mode == "chandelier":
            atr = self.cfg.atr_value or 0.0
            if self.state.peak is not None and atr > 0:
                if self.side == "buy":
                    new_stop = round(self.state.peak - (self.cfg.atr_k * atr), self.cfg.price_scale)
                else:
                    new_stop = round(self.state.peak + (self.cfg.atr_k * atr), self.cfg.price_scale)
                self.state.stop = new_stop
                self.state.last_update_ts = now
                return new_stop
            return None  # ATR or peak not ready yet

        # === Cooldown check for non-chandelier modes ===
        if not self._cooldown_ok(now):
            return None

        # === Compute candidate stop ===
        new_stop = self._candidate_stop(last)
        if new_stop is None:
            return None
        new_stop = round(new_stop, self.cfg.price_scale)

        # === Prevent regression ===
        if self.state.stop is not None:
            if self.side == "buy" and new_stop < self.state.stop:
                return None
            if self.side == "sell" and new_stop > self.state.stop:
                return None

        # === Improvement threshold (ATR-based modes only) ===
        if self.cfg.mode == "atr" and self.cfg.atr_value:
            if self.state.stop is not None:
                diff = abs(new_stop - self.state.stop)
                if diff < self.cfg.min_improve_atr * self.cfg.atr_value:
                    return None

        # === Store and return ===
        self.state.stop = new_stop
        self.state.last_update_ts = now
        return self.state.stop



    def lock(self) -> None:
        """Lock engine to prevent stop updates."""
        self.state.locked = True
