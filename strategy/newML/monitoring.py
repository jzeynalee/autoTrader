# monitoring_ml.py
from prometheus_client import Gauge, Counter, Histogram
HAS_PROM = True


class MLMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLMonitor, cls).__new__(cls)
            cls._instance._init_metrics()
        return cls._instance

    def _init_metrics(self):
        if not HAS_PROM: return
        
        self.regime_gauge = Gauge('ml_regime_current', 'Current Market Regime ID')
        self.confidence_gauge = Gauge('ml_regime_confidence', 'Model Confidence Posterior')
        self.dwell_hist = Histogram('ml_regime_dwell_time', 'Time spent in regime')
        self.fallback_counter = Counter('ml_fallback_events', 'Count of fallback triggers')
        
        self.current_regime = -1
        self.dwell_start_time = 0

    def record_inference(self, regime: int, confidence: float):
        if not HAS_PROM: return
        
        self.regime_gauge.set(regime)
        self.confidence_gauge.set(confidence)
        
        # Dwell Time Logic
        if regime != self.current_regime:
            # Regime Switch Happened
            # Record duration of previous regime (pseudo-code using simple counter)
            self.current_regime = regime
            # Reset dwell counter logic here in real implementation

    def record_fallback(self):
        if not HAS_PROM: return
        self.fallback_counter.inc()

monitor = MLMonitor()