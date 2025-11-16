
def test_swing_extraction():
    """Verify swing count meets expectations."""
    df = load_test_data('BTCUSDT_1h', days=365)
    detector = AdvancedRegimeDetectionSystem(atr_multiplier=1.0)
    
    swings = detector._extract_feature_engineered_swings(df)
    
    # Expect ~2-4 swings per week = ~100-200/year
    assert len(swings) >= 100, f"Only {len(swings)} swings detected"
    assert len(swings) <= 300, f"Too many swings: {len(swings)}"

def test_instance_segmentation():
    """Verify instance count scales with swing count."""
    df = load_test_data('BTCUSDT_1h', days=365)
    detector = AdvancedRegimeDetectionSystem()
    
    df_with_regimes = detector.detect_advanced_market_regimes(df)
    instance_count = df_with_regimes['regime_instance_id'].nunique()
    
    # Expect 100-150 instances for 1 year of hourly data
    assert instance_count >= 100, f"Only {instance_count} instances"
    assert instance_count <= 200, f"Too many instances: {instance_count}"

### 2. Integration Tests

def test_full_pipeline():
    """Test complete flow from data load to playbook generation."""
    db_connector = DatabaseConnector()
    
    # Run pipeline
    run_strategy_discovery(db_connector)
    
    # Verify results in database
    playbook = db_connector.get_strategy_playbook()
    
    assert len(playbook) >= 100, "Insufficient playbook entries"
    
    # Verify distribution across regimes
    regime_distribution = Counter([p['regime_type'] for p in playbook])
    assert len(regime_distribution) == 6, "Not all regime types represented"