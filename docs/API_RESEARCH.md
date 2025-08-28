# Wildlife Species Identification API Research

**Date**: August 27, 2025  
**Goal**: Find free wildlife identification APIs for garden camera system

## Research Summary

### ❌ iNaturalist Computer Vision API
**Status**: Not publicly available
- **Access**: Restricted to select individuals/organizations via email approval
- **Cost**: Fee-based access for research/commercial use  
- **Endpoint**: No working public endpoints found
- **Contact**: carrie@inaturalist.org for potential access
- **Rate Limits**: If available, 100 requests/minute max
- **Verdict**: Not viable for our project

### ✅ Google SpeciesNet (March 2025 Release)
**Status**: FREE and Open Source - **RECOMMENDED**
- **Accuracy**: 
  - 99.4% animal detection rate
  - 98.7% correct when animal predicted  
  - 94.5% species-level identification accuracy
- **Dataset**: 65M+ training images, 2000+ classification labels
- **License**: Apache 2.0 (allows commercial use)
- **Architecture**: EfficientNet V2 M (optimized for efficiency)
- **Access**: GitHub repository with full model and documentation
- **Target Use**: Specifically designed for camera trap images

**Technical Requirements**:
- Python environment with PyTorch
- Installation: `pip install speciesnet`
- GPU acceleration supported (NVIDIA recommended)
- Works on Linux and Windows

**Pi Zero 2 W Compatibility**:
- ⚠️ **Likely too resource-intensive** for 512MB RAM
- EfficientNet V2 M still requires significant memory
- GPU acceleration not available on Pi Zero
- **Verdict**: May need cloud deployment or fallback solution

### ⚠️ PlantNet API  
**Status**: FREE but limited scope
- **Scope**: Plants only (no animals/wildlife)
- **Rate Limits**: 500 identifications/day free
- **Accuracy**: 97% genus level, 60% species level
- **Verdict**: Not suitable for wildlife detection

### 💰 Cloud Vision APIs (Limited Free Tiers)
**Google Cloud Vision**:
- 1,000 requests/month free
- Generic object detection (not wildlife-specific)

**Azure Computer Vision**:
- 5,000 requests/month free  
- Generic object detection (not wildlife-specific)

**AWS Rekognition**:
- 5,000 images/month free
- Generic object detection (not wildlife-specific)

### 🔍 Other Options Investigated

**Roboflow Bird Detection API**:
- ~30 bird species only
- 89.7% mAP score
- Limited species coverage

**Clarifai**: 
- General-purpose vision API
- Not wildlife-specific

## Implementation Strategy

### Phase 1: Mock Implementation
- Build system with "Unknown Species" placeholder
- Implement full pipeline: motion → capture → "identification" → Telegram → database
- Validate architecture before species ID integration

### Phase 2: Species Identification Options (In Priority Order)

1. **SpeciesNet Cloud Wrapper** (Recommended)
   - Deploy SpeciesNet on cloud instance (Google Cloud, AWS, etc.)
   - Create simple REST API wrapper
   - Pi Zero calls cloud endpoint
   - **Pros**: Full SpeciesNet accuracy, manageable costs
   - **Cons**: Requires cloud deployment, ongoing costs

2. **Hybrid Approach**
   - Use cloud APIs (Google Vision, etc.) with free tier limits
   - Fall back to "Unknown Species" when limits exceeded
   - **Pros**: Simple integration, some species identification
   - **Cons**: Generic detection, limited accuracy

3. **Local Lightweight Model** 
   - Research smaller models (MobileNet, etc.)
   - Sacrifice accuracy for Pi Zero compatibility
   - **Pros**: No cloud dependency, no ongoing costs
   - **Cons**: Much lower accuracy, limited species coverage

### Phase 3: Future Improvements
- Monitor SpeciesNet for lighter model variants
- Consider Pi upgrade (Pi 4/5) for local SpeciesNet
- Evaluate edge AI solutions (Google Coral, etc.)

## Technical Constraints: Pi Zero 2 W

**Hardware Limits**:
- 512MB RAM (vs 4-8GB typical for AI models)
- Quad-core ARM Cortex-A53 @ 1GHz
- No GPU acceleration
- Limited compute resources

**Performance Expectations**:
- SpeciesNet: Likely too resource-intensive
- Lightweight models: 3-10 seconds processing time
- Cloud APIs: 1-5 seconds (network dependent)

## Recommendation

**Start with Phase 1**: Build complete system with mock species detection. This validates the entire architecture while we resolve species identification.

**Target for Phase 2**: Deploy SpeciesNet on cloud instance with REST API wrapper. This gives us the best accuracy specifically designed for wildlife camera traps while keeping Pi Zero code simple.