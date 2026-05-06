"""
Comprehensive test suite for OptionsGreeksCalculator.
Validates mathematical accuracy, edge cases, and performance.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from app.domain.options_calculator import OptionsGreeksCalculator


class TestTimeToExpiry:
    """Tests for time to expiry calculation."""
    
    def test_valid_expiry_date(self):
        """Test valid expiry date calculation."""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        T = OptionsGreeksCalculator.calculate_time_to_expiry(tomorrow)
        assert 0 < T < 2/365  # Less than 2 days
    
    def test_expiry_in_one_year(self):
        """Test expiry date one year from now."""
        one_year = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
        T = OptionsGreeksCalculator.calculate_time_to_expiry(one_year)
        assert 0.99 < T < 1.01  # Approximately 1 year
    
    def test_past_expiry_raises_error(self):
        """Test that past expiry date raises error."""
        past_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        with pytest.raises(ValueError):
            OptionsGreeksCalculator.calculate_time_to_expiry(past_date)
    
    def test_invalid_date_format(self):
        """Test invalid date format raises error."""
        with pytest.raises(ValueError):
            OptionsGreeksCalculator.calculate_time_to_expiry("2026/05/06")


class TestDelta:
    """Tests for Delta calculations."""
    
    def test_call_delta_atm(self):
        """Test ATM call delta (should be ~0.5)."""
        S, K = 100, 100
        delta = OptionsGreeksCalculator.calculate_call_delta(
            S=S, K=K, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        assert 0.4 < delta < 0.6  # ATM call delta near 0.5
    
    def test_call_delta_bounds(self):
        """Test call delta is bounded [0, 1]."""
        for S in [50, 100, 150]:
            for T in [0.1, 0.5, 1.0]:
                delta = OptionsGreeksCalculator.calculate_call_delta(
                    S=S, K=100, r=0.05, q=0.0, sigma=0.2, T=T
                )
                assert 0 <= delta <= 1
    
    def test_put_delta_bounds(self):
        """Test put delta is bounded [-1, 0]."""
        for S in [50, 100, 150]:
            for T in [0.1, 0.5, 1.0]:
                delta = OptionsGreeksCalculator.calculate_put_delta(
                    S=S, K=100, r=0.05, q=0.0, sigma=0.2, T=T
                )
                assert -1 <= delta <= 0
    
    def test_call_put_parity(self):
        """Test call-put delta parity: delta_call - delta_put = e^(-qT)."""
        S, K = 100, 100
        delta_call = OptionsGreeksCalculator.calculate_call_delta(
            S=S, K=K, r=0.05, q=0.02, sigma=0.2, T=1.0
        )
        delta_put = OptionsGreeksCalculator.calculate_put_delta(
            S=S, K=K, r=0.05, q=0.02, sigma=0.2, T=1.0
        )
        expected_diff = np.exp(-0.02 * 1.0)
        assert abs((delta_call - delta_put) - expected_diff) < 0.001


class TestGamma:
    """Tests for Gamma calculations."""
    
    def test_gamma_always_positive(self):
        """Test gamma is always positive."""
        for S in [50, 100, 150]:
            for option_type in ["call", "put"]:
                gamma = OptionsGreeksCalculator.calculate_gamma(
                    S=S, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
                )
                assert gamma > 0
    
    def test_gamma_atm_maximum(self):
        """Test gamma peaks when option is ATM."""
        gamma_atm = OptionsGreeksCalculator.calculate_gamma(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        gamma_itm = OptionsGreeksCalculator.calculate_gamma(
            S=110, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        gamma_otm = OptionsGreeksCalculator.calculate_gamma(
            S=90, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        assert gamma_atm > gamma_itm and gamma_atm > gamma_otm
    
    def test_gamma_increases_near_expiry(self):
        """Test gamma increases as expiry approaches."""
        gamma_short = OptionsGreeksCalculator.calculate_gamma(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=0.01
        )
        gamma_long = OptionsGreeksCalculator.calculate_gamma(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        assert gamma_short > gamma_long


class TestTheta:
    """Tests for Theta calculations."""
    
    def test_call_theta_negative_long(self):
        """Test long call theta is typically negative (time decay)."""
        theta = OptionsGreeksCalculator.calculate_theta(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=0.5, option_type="call"
        )
        assert theta < 0  # Long call loses value with time
    
    def test_put_theta_negative_long(self):
        """Test long put theta is typically negative (time decay)."""
        theta = OptionsGreeksCalculator.calculate_theta(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=0.5, option_type="put"
        )
        assert theta < 0  # Long put loses value with time
    
    def test_theta_magnitude_increases_near_expiry(self):
        """Test theta magnitude increases as expiry approaches."""
        theta_short = abs(OptionsGreeksCalculator.calculate_theta(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=0.01, option_type="call"
        ))
        theta_long = abs(OptionsGreeksCalculator.calculate_theta(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        ))
        assert theta_short > theta_long


class TestVega:
    """Tests for Vega calculations."""
    
    def test_vega_positive(self):
        """Test vega is always positive (higher vol = higher option value)."""
        for vol in [0.1, 0.2, 0.5]:
            vega = OptionsGreeksCalculator.calculate_vega(
                S=100, K=100, r=0.05, q=0.0, sigma=vol, T=1.0
            )
            assert vega > 0
    
    def test_vega_atm_maximum(self):
        """Test vega peaks when option is ATM."""
        vega_atm = OptionsGreeksCalculator.calculate_vega(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        vega_itm = OptionsGreeksCalculator.calculate_vega(
            S=120, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        vega_otm = OptionsGreeksCalculator.calculate_vega(
            S=80, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        assert vega_atm > vega_itm and vega_atm > vega_otm
    
    def test_vega_increases_with_maturity(self):
        """Test vega increases with time to maturity."""
        vega_short = OptionsGreeksCalculator.calculate_vega(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=0.01
        )
        vega_long = OptionsGreeksCalculator.calculate_vega(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=2.0
        )
        assert vega_long > vega_short


class TestRho:
    """Tests for Rho calculations."""
    
    def test_call_rho_positive(self):
        """Test call rho is positive (higher rates = higher call value)."""
        rho = OptionsGreeksCalculator.calculate_rho(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        )
        assert rho > 0
    
    def test_put_rho_negative(self):
        """Test put rho is negative (higher rates = lower put value)."""
        rho = OptionsGreeksCalculator.calculate_rho(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="put"
        )
        assert rho < 0
    
    def test_rho_increases_with_maturity(self):
        """Test rho increases with time to maturity."""
        rho_short = abs(OptionsGreeksCalculator.calculate_rho(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=0.1, option_type="call"
        ))
        rho_long = abs(OptionsGreeksCalculator.calculate_rho(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=2.0, option_type="call"
        ))
        assert rho_long > rho_short


class TestOptionPrice:
    """Tests for option price calculations."""
    
    def test_call_price_positive(self):
        """Test call price is always positive."""
        price = OptionsGreeksCalculator.calculate_option_price(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        )
        assert price > 0
    
    def test_call_intrinsic_value_lower_bound(self):
        """Test call price >= intrinsic value."""
        S, K = 120, 100
        price = OptionsGreeksCalculator.calculate_option_price(
            S=S, K=K, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        )
        intrinsic = max(S - K, 0)
        assert price >= intrinsic
    
    def test_put_intrinsic_value_lower_bound(self):
        """Test put price >= intrinsic value."""
        S, K = 80, 100
        price = OptionsGreeksCalculator.calculate_option_price(
            S=S, K=K, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="put"
        )
        intrinsic = max(K - S, 0)
        assert price >= intrinsic
    
    def test_atm_call_vs_put_close(self):
        """Test ATM call and put prices are close (near zero dividend)."""
        call_price = OptionsGreeksCalculator.calculate_option_price(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        )
        put_price = OptionsGreeksCalculator.calculate_option_price(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="put"
        )
        # At-the-money option prices should be similar
        assert abs(call_price - put_price) < 1.0


class TestAllGreeks:
    """Tests for the combined all_greeks calculation."""
    
    def test_all_greeks_includes_all_values(self):
        """Test all_greeks returns all required Greeks."""
        result = OptionsGreeksCalculator.calculate_all_greeks(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        )
        
        required_keys = {'delta', 'gamma', 'theta', 'vega', 'rho', 'price'}
        assert set(result.keys()) == required_keys
    
    def test_all_greeks_call_vs_individual(self):
        """Test all_greeks matches individual Greek calculations for calls."""
        params = {'S': 100, 'K': 100, 'r': 0.05, 'q': 0.0, 'sigma': 0.2, 'T': 1.0}
        
        all_result = OptionsGreeksCalculator.calculate_all_greeks(
            **params, option_type="call"
        )
        
        delta = OptionsGreeksCalculator.calculate_call_delta(**params)
        gamma = OptionsGreeksCalculator.calculate_gamma(**params)
        
        assert abs(all_result['delta'] - delta) < 1e-6
        assert abs(all_result['gamma'] - gamma) < 1e-6
    
    def test_all_greeks_put_vs_individual(self):
        """Test all_greeks matches individual Greek calculations for puts."""
        params = {'S': 100, 'K': 100, 'r': 0.05, 'q': 0.0, 'sigma': 0.2, 'T': 1.0}
        
        all_result = OptionsGreeksCalculator.calculate_all_greeks(
            **params, option_type="put"
        )
        
        delta = OptionsGreeksCalculator.calculate_put_delta(**params)
        rho = OptionsGreeksCalculator.calculate_rho(**params, option_type="put")
        
        assert abs(all_result['delta'] - delta) < 1e-6
        assert abs(all_result['rho'] - rho) < 1e-6
    
    def test_all_greeks_performance(self):
        """Test all_greeks is efficient (combined vs individual)."""
        import time
        
        params = {'S': 100, 'K': 100, 'r': 0.05, 'q': 0.0, 'sigma': 0.2, 'T': 1.0}
        
        # Combined calculation
        start = time.time()
        for _ in range(1000):
            OptionsGreeksCalculator.calculate_all_greeks(**params, option_type="call")
        combined_time = time.time() - start
        
        # Individual calculations
        start = time.time()
        for _ in range(1000):
            OptionsGreeksCalculator.calculate_call_delta(**params)
            OptionsGreeksCalculator.calculate_gamma(**params)
            OptionsGreeksCalculator.calculate_theta(**params, option_type="call")
            OptionsGreeksCalculator.calculate_vega(**params)
            OptionsGreeksCalculator.calculate_rho(**params, option_type="call")
            OptionsGreeksCalculator.calculate_option_price(**params, option_type="call")
        individual_time = time.time() - start
        
        # Combined should be faster
        assert combined_time < individual_time


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_high_volatility(self):
        """Test calculations with very high volatility."""
        result = OptionsGreeksCalculator.calculate_all_greeks(
            S=100, K=100, r=0.05, q=0.0, sigma=2.0, T=1.0, option_type="call"
        )
        assert all(not np.isnan(v) and not np.isinf(v) for v in result.values())
    
    def test_very_low_volatility(self):
        """Test calculations with very low volatility."""
        result = OptionsGreeksCalculator.calculate_all_greeks(
            S=100, K=100, r=0.05, q=0.0, sigma=0.01, T=1.0, option_type="call"
        )
        assert all(not np.isnan(v) and not np.isinf(v) for v in result.values())
    
    def test_deep_itm_call(self):
        """Test deep in-the-money call."""
        result = OptionsGreeksCalculator.calculate_all_greeks(
            S=200, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        )
        assert result['delta'] > 0.9
    
    def test_deep_otm_call(self):
        """Test deep out-of-the-money call."""
        result = OptionsGreeksCalculator.calculate_all_greeks(
            S=50, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call"
        )
        assert result['delta'] < 0.1
    
    def test_dividend_yield_impact(self):
        """Test dividend yield affects delta."""
        delta_no_div = OptionsGreeksCalculator.calculate_call_delta(
            S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0
        )
        delta_with_div = OptionsGreeksCalculator.calculate_call_delta(
            S=100, K=100, r=0.05, q=0.05, sigma=0.2, T=1.0
        )
        assert delta_no_div > delta_with_div  # Dividends reduce call delta
