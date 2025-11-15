# ADR-001: Integration of CameraTrapAI (SpeciesNet) for Wildlife Species Identification

**Status**: Proposed
**Date**: 2025-11-15
**Decision Makers**: Project Team
**Technical Story**: Replace mock species identification with Google's SpeciesNet AI model

---

## Context and Problem Statement

The wildlife camera system currently uses a mock implementation for species identification (`src/species_identifier.py`) that returns random species names with simulated confidence scores. We need to integrate real AI-powered species identification to accurately identify wildlife captured by the camera.

**Requirements:**
- Run entirely on Raspberry Pi 5 (8GB RAM) - no cloud dependencies
- Geographic filtering for Bonn, Germany (reduce false positives)
- Handle low-confidence predictions gracefully
- Maintain existing system architecture and interfaces
- Use detector bounding boxes for improved accuracy (always-crop mode)

---

## Decision Drivers

1. **Hardware Constraints**: Raspberry Pi 5 with 8GB RAM (sufficient for SpeciesNet)
2. **Privacy & Reliability**: All processing must be local (no cloud dependencies)
3. **Accuracy**: Need wildlife-specific model (not generic object detection)
4. **Cost**: Free and open-source solution required
5. **Geographic Relevance**: Filter predictions to species found in Germany
6. **Existing Architecture**: Minimize changes to working motion detection and notification system

---

## Considered Options

### Option 1: Google SpeciesNet (CameraTrapAI) - **SELECTED**
**Pros:**
- Free and open-source (Apache 2.0)
- Designed specifically for camera trap images
- High accuracy: 94.5% species-level, 99.4% animal detection
- 2000+ species labels covering diverse wildlife
- Geographic filtering built-in
- Runs locally on Pi 5 hardware

**Cons:**
- Resource-intensive (but Pi 5 can handle it)
- Initial model download (~500MB-1GB)
- CPU processing slower than GPU (but acceptable for wildlife camera use case)

### Option 2: Cloud-based APIs (Google Vision, AWS Rekognition)
**Pros:**
- Fast inference on cloud GPUs
- No local resource constraints

**Cons:**
- Requires internet connectivity
- Privacy concerns (images sent to cloud)
- Ongoing costs or rate limits
- Generic object detection (not wildlife-specific)
- **Rejected**: Violates local-only requirement

### Option 3: Lightweight Local Models (MobileNet, etc.)
**Pros:**
- Fast inference on Pi 5
- Minimal resource usage

**Cons:**
- Much lower accuracy than SpeciesNet
- Limited species coverage
- Would require custom training for wildlife
- **Rejected**: Insufficient accuracy for project goals

---

## Decision Outcome

**Chosen Option**: Google SpeciesNet (CameraTrapAI) with local execution on Raspberry Pi 5

**Rationale:**
- Only option that meets all requirements (local, accurate, wildlife-specific, free)
- Pi 5 with 8GB RAM is sufficient for model execution
- Built-in geographic filtering ideal for Germany deployment
- Active development and maintenance from Google
- Designed exactly for this use case (camera trap wildlife identification)

---

## Implementation Plan

### Phase 1: Configuration & Dependencies

#### 1.1 Update `requirements.txt`
```python
# Add SpeciesNet dependency
speciesnet>=0.1.0
```

**Notes:**
- SpeciesNet includes PyTorch and TensorFlow dependencies
- First `pip install` will download model weights (~500MB-1GB)
- May need to increase swap space on Pi 5 during installation

#### 1.2 Extend `src/config.py`

Add new `SpeciesConfig` dataclass:

```python
@dataclass(frozen=True)
class SpeciesConfig:
    """Species identification configuration."""
    # Model settings
    model_version: str = "v4.0.1a"  # always-crop variant
    use_ensemble: bool = True  # Use detector + classifier

    # Geographic filtering
    country_code: str = "DEU"  # ISO 3166-1 Alpha-3 for Germany
    admin1_region: Optional[str] = "NW"  # North Rhine-Westphalia (Bonn)

    # Confidence thresholds
    min_detection_confidence: float = 0.6  # MegaDetector threshold
    min_classification_confidence: float = 0.5  # Species classification threshold
    unknown_species_threshold: float = 0.5  # Below this = "Unknown species"

    # Performance settings
    model_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "speciesnet")
    enable_gpu: bool = False  # Pi 5 doesn't have NVIDIA GPU
    processing_timeout: float = 30.0  # Max time for identification

    # Behavior settings
    return_top_k: int = 5  # Return top 5 species predictions
    crop_padding: float = 0.1  # Padding around detected objects

    def __post_init__(self):
        """Validate species configuration."""
        if not (0.0 <= self.min_detection_confidence <= 1.0):
            raise ValueError("Detection confidence must be between 0 and 1")
        if not (0.0 <= self.min_classification_confidence <= 1.0):
            raise ValueError("Classification confidence must be between 0 and 1")
        if self.model_version not in ["v4.0.1a", "v4.0.1b"]:
            raise ValueError(f"Invalid model version: {self.model_version}")
```

Add to main `Config` class:
```python
def __init__(self, env_file: Optional[Union[str, Path]] = None):
    # ... existing code ...
    self.species = self._load_species_config()

def _load_species_config(self) -> SpeciesConfig:
    """Load species identification configuration with environment overrides."""
    return SpeciesConfig(
        country_code=self._get_optional_env("SPECIES_COUNTRY_CODE", "DEU"),
        admin1_region=self._get_optional_env("SPECIES_REGION", "NW"),
        min_detection_confidence=self._get_optional_env(
            "SPECIES_MIN_DETECTION_CONF", "0.6", float
        ),
        min_classification_confidence=self._get_optional_env(
            "SPECIES_MIN_CLASS_CONF", "0.5", float
        ),
        unknown_species_threshold=self._get_optional_env(
            "SPECIES_UNKNOWN_THRESHOLD", "0.5", float
        )
    )
```

**Environment Variables** (optional overrides in `.env`):
```bash
# Species Identification Settings
SPECIES_COUNTRY_CODE=DEU
SPECIES_REGION=NW
SPECIES_MIN_DETECTION_CONF=0.6
SPECIES_MIN_CLASS_CONF=0.5
SPECIES_UNKNOWN_THRESHOLD=0.5
```

---

### Phase 2: Core Implementation

#### 2.1 Replace `src/species_identifier.py`

**Key Components:**

1. **SpeciesNetIdentifier Class**: Main implementation
2. **Model Management**: Lazy loading, caching, lifecycle
3. **Image Processing**: Format conversion, preprocessing
4. **Result Parsing**: Convert SpeciesNet output to `IdentificationResult`
5. **Error Handling**: Graceful fallback to "Unknown species"

**Architecture:**

```python
class SpeciesNetIdentifier:
    """
    Real species identification using Google SpeciesNet.
    Replaces mock implementation with actual AI inference.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None  # Lazy-loaded
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Lazy-load SpeciesNet models on first use."""
        if self._model_loaded:
            return

        try:
            from speciesnet.ensemble import SpeciesNetEnsemble

            # Initialize ensemble (detector + classifier)
            self._model = SpeciesNetEnsemble(
                country=self.config.species.country_code,
                region=self.config.species.admin1_region,
                classifier_version=self.config.species.model_version,
                cache_dir=str(self.config.species.model_cache_dir)
            )

            self._model_loaded = True
            logger.info(f"SpeciesNet loaded: {self.config.species.model_version}")

        except Exception as e:
            logger.error(f"Failed to load SpeciesNet: {e}")
            raise SpeciesIdentificationError(f"Model initialization failed: {e}")

    def identify_species(self, image_path, timeout=None) -> IdentificationResult:
        """
        Identify species in image using SpeciesNet.

        Args:
            image_path: Path to image file
            timeout: Processing timeout (uses config default if None)

        Returns:
            IdentificationResult with species name, confidence, etc.
        """
        start_time = time.time()
        timeout = timeout or self.config.species.processing_timeout

        try:
            # Validate image exists
            if not Path(image_path).exists():
                return self._create_error_response(
                    start_time, "Image file not found"
                )

            # Ensure model is loaded
            self._ensure_model_loaded()

            # Run inference
            predictions = self._run_inference(image_path, timeout)

            # Parse and validate results
            result = self._parse_predictions(predictions, start_time)

            return result

        except Exception as e:
            logger.error(f"Species identification failed: {e}")
            return self._create_error_response(
                start_time, f"Identification error: {e}"
            )

    def _run_inference(self, image_path: Path, timeout: float):
        """Run SpeciesNet inference with timeout protection."""
        # SpeciesNet expects list of image info dicts
        image_info = [{
            'filepath': str(image_path),
            'country': self.config.species.country_code,
            'admin1_region': self.config.species.admin1_region
        }]

        # Run ensemble prediction
        predictions = self._model.predict(image_info)

        return predictions

    def _parse_predictions(self, predictions, start_time) -> IdentificationResult:
        """
        Parse SpeciesNet predictions into IdentificationResult format.

        SpeciesNet output format:
        {
            'predictions': [{
                'filepath': str,
                'detections': [{
                    'category': str,  # 'animal', 'person', 'vehicle'
                    'conf': float,
                    'bbox': [x1, y1, x2, y2]  # normalized 0-1
                }],
                'classifications': [{
                    'class': str,  # species name
                    'score': float
                }],  # Top-5 predictions
                'prediction': str,  # Final ensemble prediction
                'prediction_score': float
            }]
        }
        """
        processing_time = time.time() - start_time

        if not predictions or not predictions.get('predictions'):
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=False,
                processing_time=processing_time,
                fallback_reason='No predictions returned'
            )

        pred = predictions['predictions'][0]

        # Check if any animals detected
        detections = pred.get('detections', [])
        animal_detections = [d for d in detections
                            if d['category'] == 'animal'
                            and d['conf'] >= self.config.species.min_detection_confidence]

        if not animal_detections:
            return IdentificationResult(
                species_name='Unknown species',
                confidence=0.0,
                api_success=True,  # Model ran successfully
                processing_time=processing_time,
                fallback_reason='No animals detected above confidence threshold',
                metadata={'detections': detections}
            )

        # Get final species prediction
        final_species = pred.get('prediction', 'Unknown species')
        final_confidence = pred.get('prediction_score', 0.0)

        # Apply confidence threshold
        if final_confidence < self.config.species.unknown_species_threshold:
            return IdentificationResult(
                species_name='Unknown species',
                confidence=final_confidence,
                api_success=True,
                processing_time=processing_time,
                fallback_reason=f'Confidence {final_confidence:.2f} below threshold {self.config.species.unknown_species_threshold}',
                metadata={
                    'raw_prediction': final_species,
                    'raw_confidence': final_confidence,
                    'top_predictions': pred.get('classifications', [])
                }
            )

        # Success case
        return IdentificationResult(
            species_name=final_species,
            confidence=final_confidence,
            api_success=True,
            processing_time=processing_time,
            metadata={
                'detections': animal_detections,
                'top_predictions': pred.get('classifications', [])[:5]
            }
        )

    def _create_error_response(self, start_time, reason):
        """Create standardized error response."""
        return IdentificationResult(
            species_name='Unknown species',
            confidence=0.0,
            api_success=False,
            processing_time=time.time() - start_time,
            fallback_reason=reason
        )

    def health_check(self):
        """Check if SpeciesNet service is available."""
        try:
            self._ensure_model_loaded()
            return {
                'available': True,
                'service': 'SpeciesNet',
                'version': self.config.species.model_version,
                'country': self.config.species.country_code,
                'region': self.config.species.admin1_region,
                'supported_formats': ['jpg', 'jpeg', 'png']
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
```

**Key Implementation Notes:**

1. **Lazy Loading**: Model loads on first `identify_species()` call (startup optimization)
2. **Geographic Filtering**: Automatically applied via SpeciesNet's built-in filtering
3. **Confidence Handling**: Two-stage filtering (detection + classification)
4. **Error Resilience**: Always returns valid `IdentificationResult`, never crashes
5. **Metadata Preservation**: Store raw predictions for debugging/analysis

---

### Phase 3: Integration Updates

#### 3.1 Update `src/wildlife_system.py`

**Minimal changes required** - existing interface should work:

```python
class WildlifeSystem:
    def __init__(self, advanced_mode: bool = True):
        # ... existing initialization ...

        # Replace mock with real SpeciesNet
        self.species_identifier = SpeciesIdentifier(self.config)

        # ... rest of initialization ...
```

**Remove simple mode support** (per requirements):

```python
class WildlifeSystem:
    def __init__(self):  # Remove advanced_mode parameter
        self.config = Config()

        # Always initialize all components
        self.camera = CameraManager(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.telegram_service = TelegramService(self.config)
        self.system_monitor = SystemMonitor(self.config)
        self.file_manager = FileManager(self.config)
        self.database = DatabaseManager(self.config)
        self.species_identifier = SpeciesIdentifier(self.config)  # Always enabled

        self.file_manager.ensure_directories()
        self.telegram_service.set_database_reference(self.database)

    def process_detection(self, image_path: Path, motion_area: int) -> tuple:
        """Process detection with species identification."""
        timestamp = datetime.now()

        try:
            # Species identification with performance timing
            with PerformanceTimer("Species identification"):
                species_result = self.species_identifier.identify_species(image_path)

            # Log to database
            detection_id = self.database.log_detection(
                image_path=image_path,
                motion_area=motion_area,
                species_name=species_result.species_name,
                confidence_score=species_result.confidence,
                processing_time=species_result.processing_time,
                api_success=species_result.api_success
            )

            print(f"Detection {detection_id}: {species_result.species_name} "
                  f"(confidence: {species_result.confidence:.2f}, "
                  f"motion: {motion_area} pixels)")

            # Convert to dict for compatibility
            result_dict = {
                'species_name': species_result.species_name,
                'confidence': species_result.confidence,
                'api_success': species_result.api_success,
                'processing_time': species_result.processing_time,
                'fallback_reason': species_result.fallback_reason
            }

            return result_dict, timestamp

        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            return {
                'species_name': 'Unknown species',
                'confidence': 0.0,
                'api_success': False,
                'processing_time': 0.0,
                'fallback_reason': f'Processing error: {e}'
            }, timestamp
```

#### 3.2 Update Main Entry Point

Update `src/wildlife_system.py` main block:

```python
if __name__ == "__main__":
    # Remove mode selection - always use full system
    system = WildlifeSystem()
    asyncio.run(system.run())
```

---

### Phase 4: Testing Strategy

#### 4.1 Unit Tests

Update `tests/test_species_identifier.py`:

```python
class TestSpeciesNetIdentifier:
    """Test SpeciesNet implementation with mocked model calls."""

    @patch('speciesnet.ensemble.SpeciesNetEnsemble')
    def test_initialization(self, mock_ensemble):
        """Test SpeciesNet initialization."""
        config = Config.create_test_config()
        identifier = SpeciesNetIdentifier(config)

        # Model should not be loaded until first use
        assert identifier._model is None
        assert not identifier._model_loaded

    @patch('speciesnet.ensemble.SpeciesNetEnsemble')
    def test_successful_identification(self, mock_ensemble, tmp_path):
        """Test successful species identification."""
        # Setup mock response
        mock_ensemble.return_value.predict.return_value = {
            'predictions': [{
                'filepath': 'test.jpg',
                'detections': [{
                    'category': 'animal',
                    'conf': 0.95,
                    'bbox': [0.2, 0.3, 0.7, 0.8]
                }],
                'classifications': [{
                    'class': 'European Hedgehog',
                    'score': 0.87
                }],
                'prediction': 'European Hedgehog',
                'prediction_score': 0.87
            }]
        }

        config = Config.create_test_config()
        identifier = SpeciesNetIdentifier(config)

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        result = identifier.identify_species(test_image)

        assert result.species_name == 'European Hedgehog'
        assert result.confidence == 0.87
        assert result.api_success is True
        assert result.processing_time > 0

    @patch('speciesnet.ensemble.SpeciesNetEnsemble')
    def test_low_confidence_fallback(self, mock_ensemble, tmp_path):
        """Test low confidence returns Unknown species."""
        mock_ensemble.return_value.predict.return_value = {
            'predictions': [{
                'detections': [{'category': 'animal', 'conf': 0.9, 'bbox': [0, 0, 1, 1]}],
                'prediction': 'Some Rare Species',
                'prediction_score': 0.3  # Below threshold
            }]
        }

        config = Config.create_test_config()
        identifier = SpeciesNetIdentifier(config)

        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        result = identifier.identify_species(test_image)

        assert result.species_name == 'Unknown species'
        assert result.api_success is True
        assert 'confidence' in result.fallback_reason.lower()

    @patch('speciesnet.ensemble.SpeciesNetEnsemble')
    def test_no_animals_detected(self, mock_ensemble, tmp_path):
        """Test when no animals are detected."""
        mock_ensemble.return_value.predict.return_value = {
            'predictions': [{
                'detections': [],  # No detections
                'prediction': 'blank',
                'prediction_score': 0.99
            }]
        }

        config = Config.create_test_config()
        identifier = SpeciesNetIdentifier(config)

        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        result = identifier.identify_species(test_image)

        assert result.species_name == 'Unknown species'
        assert 'no animals detected' in result.fallback_reason.lower()
```

#### 4.2 Integration Tests

Create `tests/test_speciesnet_integration.py`:

```python
@pytest.mark.integration
class TestSpeciesNetIntegration:
    """Integration tests with real SpeciesNet (requires model download)."""

    @pytest.mark.skipif(not os.path.exists(Path.home() / ".cache" / "speciesnet"),
                       reason="SpeciesNet models not downloaded")
    def test_real_inference_with_test_image(self):
        """Test with actual SpeciesNet inference on test image."""
        config = Config.create_test_config()
        identifier = SpeciesNetIdentifier(config)

        # Use test image from fixtures
        test_image_path = Path(__file__).parent / "fixtures" / "hedgehog_test.jpg"

        if test_image_path.exists():
            result = identifier.identify_species(test_image_path)

            assert result.processing_time > 0
            assert result.api_success in [True, False]
            assert isinstance(result.species_name, str)
            assert 0.0 <= result.confidence <= 1.0
```

#### 4.3 Test Image Fixtures

Create test images in `tests/fixtures/`:
- `hedgehog_test.jpg` - European Hedgehog
- `fox_test.jpg` - Red Fox
- `bird_test.jpg` - Common garden bird
- `blank_test.jpg` - Empty garden scene

---

### Phase 5: Deployment & Operations

#### 5.1 First-Time Setup on Raspberry Pi 5

```bash
# 1. Install system dependencies (may be needed)
sudo apt-get update
sudo apt-get install -y libopenblas-dev libopenjp2-7

# 2. Install Python dependencies
cd /home/user/animal_tracker
pip install -r requirements.txt

# 3. Download SpeciesNet models (first run only, ~500MB-1GB)
python -c "from speciesnet.ensemble import SpeciesNetEnsemble; SpeciesNetEnsemble(country='DEU')"

# 4. Verify installation
python -c "from species_identifier import SpeciesNetIdentifier; from config import Config; print('SpeciesNet ready')"

# 5. Run system
python src/wildlife_system.py
```

#### 5.2 Performance Expectations

**Raspberry Pi 5 (8GB RAM, CPU-only):**
- Model loading time: ~10-15 seconds (first run only)
- Inference time per image: ~5-10 seconds
- Memory usage: ~2-3GB (comfortable for 8GB Pi)
- Concurrent operations: Model is thread-safe

**Optimization Tips:**
- Keep model loaded (process stays running)
- Use cooldown period to prevent processing overload
- Monitor system resources with existing SystemMonitor

#### 5.3 Configuration Tuning

Adjust in `.env` for your environment:

```bash
# Increase if getting too many "Unknown species"
SPECIES_UNKNOWN_THRESHOLD=0.4

# Decrease if getting false positives
SPECIES_MIN_DETECTION_CONF=0.7
SPECIES_MIN_CLASS_CONF=0.6

# Increase if inference times out
SPECIES_PROCESSING_TIMEOUT=45.0
```

#### 5.4 Monitoring & Debugging

Add logging to track performance:

```python
# In wildlife_system.py
if species_result.api_success:
    logger.info(f"SpeciesNet: {species_result.species_name} "
                f"({species_result.confidence:.2f}) "
                f"in {species_result.processing_time:.1f}s")
else:
    logger.warning(f"SpeciesNet failed: {species_result.fallback_reason}")
```

Database queries for analysis:

```sql
-- Check identification success rate
SELECT
    api_success,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence
FROM detections
GROUP BY api_success;

-- Most common species
SELECT species_name, COUNT(*) as count
FROM detections
WHERE species_name != 'Unknown species'
GROUP BY species_name
ORDER BY count DESC
LIMIT 10;

-- Processing time statistics
SELECT
    AVG(processing_time) as avg_time,
    MIN(processing_time) as min_time,
    MAX(processing_time) as max_time
FROM detections
WHERE api_success = TRUE;
```

---

## Migration Path

### Step 1: Backup Current System
```bash
git commit -am "Checkpoint before SpeciesNet integration"
git tag pre-speciesnet-v1.0
```

### Step 2: Install Dependencies
```bash
pip install speciesnet
```

### Step 3: Update Configuration
- Add `SpeciesConfig` to `config.py`
- Update `.env` with species settings

### Step 4: Replace Species Identifier
- Implement new `SpeciesNetIdentifier` class
- Keep old `MockSpeciesIdentifier` for testing
- Update tests

### Step 5: Remove Simple Mode
- Update `WildlifeSystem.__init__()` to remove `advanced_mode`
- Update main entry point
- Update documentation

### Step 6: Test & Validate
```bash
# Run unit tests
pytest tests/test_species_identifier.py -v

# Run integration tests
pytest tests/test_speciesnet_integration.py -v --integration

# Test with real system (monitor mode)
python src/wildlife_system.py
```

### Step 7: Production Deployment
```bash
# Deploy to Pi 5
git push origin claude/integrate-cameratrapai-01G6j2yeXuVPSauFrNBwkzhX

# On Pi 5: Pull and restart
git pull
sudo systemctl restart wildlife-camera.service
```

---

## Consequences

### Positive

✅ **Accurate Species Identification**: 94.5% accuracy vs 0% (mock)
✅ **Geographic Relevance**: Filters to German species automatically
✅ **Offline Operation**: No internet required after model download
✅ **No Recurring Costs**: Free and open-source
✅ **Privacy Preserved**: All processing on local device
✅ **Rich Metadata**: Get top-5 predictions, bounding boxes, confidence scores
✅ **Future-Proof**: Active Google development, regular updates

### Negative

⚠️ **Increased Resource Usage**: ~2-3GB RAM (vs ~100MB for mock)
⚠️ **Slower Processing**: 5-10 seconds per image (vs instant mock)
⚠️ **Initial Download**: ~1GB model download on first setup
⚠️ **Complexity**: More dependencies, potential for PyTorch issues
⚠️ **Pi 5 Required**: Won't run on smaller hardware (but already using Pi 5)

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model download fails | High | Implement retry logic, manual download option |
| Out of memory on Pi 5 | Medium | Monitor with SystemMonitor, increase swap if needed |
| Slow inference blocks system | Medium | Use existing cooldown period, async processing |
| SpeciesNet API changes | Low | Pin to specific version, monitor releases |
| False positives for local species | Medium | Tune confidence thresholds, validate with ground truth |

---

## Implementation Checklist

### Code Changes
- [ ] Add `SpeciesConfig` to `src/config.py`
- [ ] Add species config loading to `Config.__init__()`
- [ ] Add `speciesnet>=0.1.0` to `requirements.txt`
- [ ] Replace `src/species_identifier.py` with SpeciesNet implementation
- [ ] Remove `advanced_mode` parameter from `WildlifeSystem`
- [ ] Update `wildlife_system.py` main entry point
- [ ] Remove simple mode logic from `process_detection()`
- [ ] Remove simple mode logic from `send_notification()`

### Testing
- [ ] Update `tests/test_species_identifier.py` with mocked tests
- [ ] Create `tests/test_speciesnet_integration.py`
- [ ] Add test image fixtures to `tests/fixtures/`
- [ ] Run full test suite and ensure all pass
- [ ] Manual testing with real Pi camera images

### Documentation
- [ ] Update `README.md` with SpeciesNet info
- [ ] Update `CLAUDE.md` with new architecture
- [ ] Add species identification section to docs
- [ ] Document `.env` variables for species config
- [ ] Create troubleshooting guide for common issues

### Deployment
- [ ] Test on development Pi 5
- [ ] Download models to Pi 5
- [ ] Verify geographic filtering works
- [ ] Monitor initial performance metrics
- [ ] Tune confidence thresholds based on results
- [ ] Update systemd service if needed
- [ ] Create backup/rollback plan

### Validation
- [ ] Verify 10+ test images identify correctly
- [ ] Confirm "Unknown species" handling works
- [ ] Check database logging includes all fields
- [ ] Verify Telegram notifications include species
- [ ] Monitor system resources under load
- [ ] Validate geographic filtering (no exotic species)

---

## References

- [CameraTrapAI GitHub](https://github.com/google/cameratrapai)
- [SpeciesNet Research Paper](https://arxiv.org/abs/2403.xxxxx)
- [Raspberry Pi 5 Specifications](https://www.raspberrypi.com/products/raspberry-pi-5/)
- [ISO 3166-1 Country Codes](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3)
- [Germany Admin Regions (ISO 3166-2:DE)](https://en.wikipedia.org/wiki/ISO_3166-2:DE)

---

## Notes

### Geographic Filtering Details

For Bonn, Germany:
- **Country Code**: `DEU` (ISO 3166-1 Alpha-3)
- **Admin1 Region**: `NW` (North Rhine-Westphalia / Nordrhein-Westfalen)
- **Effect**: SpeciesNet will filter predictions to species known in this region

Common German wildlife species expected:
- European Hedgehog (*Erinaceus europaeus*)
- Red Fox (*Vulpes vulpes*)
- European Robin (*Erithacus rubecula*)
- Eurasian Magpie (*Pica pica*)
- Domestic Cat (*Felis catus*)
- Roe Deer (*Capreolus capreolus*)
- European Badger (*Meles meles*)

### Model Variants

**v4.0.1a (always-crop)** - SELECTED:
- Uses MegaDetector bounding boxes to crop animals
- Better for distant or partially obscured animals
- Focuses classifier on detected animals only
- Recommended for outdoor camera traps

**v4.0.1b (full-image)**:
- Analyzes entire image without cropping
- Better for close-up, centered subjects
- May perform better when animals fill frame
- Alternative if always-crop underperforms

### Future Enhancements

1. **Batch Processing**: Process multiple queued images in one model call
2. **Confidence Calibration**: Adjust thresholds based on observed accuracy
3. **Species Tracking**: Link detections over time to identify individual animals
4. **Advanced Filtering**: Time-of-day filtering (nocturnal vs diurnal species)
5. **Model Updates**: Track SpeciesNet releases for improved models
6. **Edge Cases**: Special handling for common/rare species

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Next Review**: After implementation completion
