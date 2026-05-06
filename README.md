# Real-Time Options Risk & Greek Calculator

A high-performance microservice for calculating options Greeks (Delta, Gamma, Theta, Vega, Rho) using the Black-Scholes-Merton model.

## Features

✨ **Performance**
- Sub-millisecond single option calculations
- Batch processing for 1000+ options concurrently
- Redis caching for <1ms lookups
- Vectorized NumPy calculations

🏗️ **Architecture**
- Clean separation between domain logic and API layer
- Async/await for non-blocking I/O
- Dependency injection pattern
- Multi-stage Docker build

📊 **Monitoring**
- Prometheus metrics instrumentation
- Request latency tracking
- Cache hit rate monitoring
- Redis connection status

🧪 **Testing**
- 90+ comprehensive tests
- Unit tests with mathematical validation
- Integration tests for API endpoints
- Benchmark tests against known Black-Scholes values

## Quick Start

### Docker Compose (Recommended)
```bash
docker-compose up --build
```

Access:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest -v --cov=app

# Start server
uvicorn app.main:app --reload
```

## API Endpoints

### Single Option Calculation
```bash
POST /api/v1/calculate

Request:
{
  "instrument": {
    "ticker": "AAPL",
    "option_type": "call",
    "strike_price": 150.0,
    "spot_price": 150.0,
    "expiry_date": "2026-06-06",
    "volatility": 0.20,
    "risk_free_rate": 0.05,
    "dividend_yield": 0.0
  }
}

Response:
{
  "ticker": "AAPL",
  "option_type": "call",
  "strike_price": 150.0,
  "greeks": {
    "delta": 0.6368,
    "gamma": 0.0051,
    "theta": -0.0252,
    "vega": 0.1234,
    "rho": 0.5678,
    "option_price": 7.45,
    "cache_hit": false
  },
  "calculated_at": "2026-05-06T12:34:56.123456"
}
```

### Batch Calculation
```bash
POST /api/v1/batch

Request:
{
  "instruments": [
    { /* instrument 1 */ },
    { /* instrument 2 */ },
    ...
  ]
}

Response:
{
  "results": [ /* array of calculation results */ ],
  "batch_size": 100,
  "calculation_time_ms": 45.23,
  "cache_hit_count": 23
}
```

### Health Check
```bash
GET /api/v1/health

Response:
{
  "status": "healthy",
  "redis_connected": true,
  "timestamp": "2026-05-06T12:34:56.123456"
}
```

### Prometheus Metrics
```bash
GET /metrics
```

## Performance Benchmarks

| Scenario | Latency |
|----------|---------|
| Single option (cache miss) | < 5ms |
| Single option (cache hit) | < 1ms |
| Batch 100 options | < 50ms |
| Batch 1000 options | < 500ms |

## Architecture

### Domain Layer (`app/domain/`)
- `models.py`: Pydantic validation models
- `options_calculator.py`: Black-Scholes-Merton implementation

### API Layer (`app/api/`)
- `routes.py`: FastAPI endpoints
- `dependencies.py`: Dependency injection

### Infrastructure Layer (`app/infrastructure/`)
- `cache.py`: Redis async cache manager
- `metrics.py`: Prometheus instrumentation

## Greeks Explained

- **Delta (Δ)**: Rate of change of option price with respect to underlying price
- **Gamma (Γ)**: Rate of change of delta (curvature of delta)
- **Theta (Θ)**: Time decay per day
- **Vega (ν)**: Sensitivity to changes in volatility (per 1% change)
- **Rho (ρ)**: Sensitivity to changes in interest rates (per 1% change)

## Testing

```bash
# Run all tests with coverage
pytest -v --cov=app

# Run specific test file
pytest tests/test_options_calculator.py -v

# Run with detailed output
pytest -vv --tb=short
```

## Deployment

### Docker
```bash
# Build image
docker build -t options-calculator:latest .

# Run container
docker run -p 8000:8000 \
  -e REDIS_URL=redis://localhost:6379 \
  options-calculator:latest
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## Configuration

Environment variables:
- `REDIS_URL`: Redis connection URL (default: `redis://localhost:6379`)
- `LOG_LEVEL`: Logging level (default: `info`)
- `WORKERS`: Number of Uvicorn workers (default: `4`)

## Mathematical Model

The calculator implements the **Black-Scholes-Merton model** with dividend yield support:

```
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

Where:
  S = Spot price
  K = Strike price
  r = Risk-free rate
  q = Dividend yield
  σ = Volatility
  T = Time to expiration (years)
  N() = Cumulative normal distribution
```

## Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest --cov=app`
5. Submit a pull request

## References

- Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.)
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities
- Merton, R. C. (1973). Theory of rational option pricing

## License

MIT License

## Author

Abhay Bhuva

---

Built with ❤️ for Bank of America's quantitative trading platform
