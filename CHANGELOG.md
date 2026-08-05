# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-04

### Added
- Voice reflex gate with three-tier matching (exact, fuzzy, no-match)
- Fuzzy matching engine with configurable thresholds
- Context-aware caching (context vectors influence match keys)
- Learner module for pattern acquisition over time
- Matcher module with tiered confidence routing
- Cache module with TTL and eviction
- Gate module orchestrating the full pipeline
- Comprehensive test suite: 104 tests covering all modules
- README with architecture diagram

### Technical Details
- **Tier 1 (Exact):** Hash-based lookup, confidence ≥ 0.95, zero-latency
- **Tier 2 (Fuzzy):** Weighted fuzzy matching, confidence 0.60–0.94
- **Tier 3 (No match):** Cascade to model invocation
- Context vectors influence match keys for situational awareness
- Custom fuzzy threshold configurable per deployment
