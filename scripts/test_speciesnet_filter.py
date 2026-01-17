#!/usr/bin/env python3
"""
Test script to run SpeciesNet on all images and evaluate detection performance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from species_identifier import SpeciesIdentifier
from config import Config


def main():
    # Initialize SpeciesNet
    print("=" * 80)
    print("SPECIESNET DETECTION TEST")
    print("=" * 80)
    print("\nInitializing SpeciesNet (first run downloads ~214MB model)...")

    config = Config()
    identifier = SpeciesIdentifier(config)

    # Directories to analyze
    dirs_to_test = [
        ("data/animals", "GROUND TRUTH: Animals (should detect)"),
        ("data/background", "GROUND TRUTH: Background (should NOT detect)"),
        #("data/images", "ALL IMAGES (unlabeled)"),
    ]

    all_results = {}

    for dir_path, description in dirs_to_test:
        full_path = Path("/home/daniel/animal_tracker") / dir_path
        if not full_path.exists():
            print(f"\nSkipping {dir_path} - directory not found")
            continue

        print(f"\n{'=' * 80}")
        print(f"{description}")
        print(f"{'=' * 80}")

        # Get all jpg images (not annotated ones)
        images = sorted([
            p for p in full_path.glob("*.jpg")
            if "_annotated" not in p.name
        ])

        if not images:
            print("No images found")
            continue

        print(f"Found {len(images)} images\n")

        results = []
        for img_path in images:
            print(f"Processing: {img_path.name}...", end=" ", flush=True)

            result = identifier.identify_species(str(img_path))

            # Determine if detection passed
            has_detection = (
                result.species_name != "Unknown species" and
                result.confidence > 0
            )

            # Get detector confidence if available
            detector_conf = None
            if result.detection_result and result.detection_result.detections:
                detections = result.detection_result.detections
                if detections:
                    # Get max detector confidence
                    detector_conf = max(d.get('conf', d.get('confidence', 0)) for d in detections)

            results.append({
                'image': img_path.name,
                'species': result.species_name,
                'confidence': result.confidence,
                'detector_confidence': detector_conf,
                'has_detection': has_detection,
            })

            # Print result
            if has_detection:
                det_str = f"det={detector_conf:.2f}" if detector_conf else "det=?"
                print(f"✓ {result.species_name} ({result.confidence:.2f}, {det_str})")
            else:
                det_str = f"det={detector_conf:.2f}" if detector_conf else "det=none"
                print(f"✗ No detection ({det_str})")

        all_results[dir_path] = results

        # Summary for this directory
        detected = sum(1 for r in results if r['has_detection'])
        print(f"\nSummary: {detected}/{len(results)} images with detections")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    if "data/animals" in all_results:
        animals = all_results["data/animals"]
        detected = sum(1 for r in animals if r['has_detection'])
        print(f"\nAnimals (ground truth): {detected}/{len(animals)} detected")
        print(f"  Recall: {100*detected/len(animals):.0f}%")
        if detected < len(animals):
            print("  MISSED:")
            for r in animals:
                if not r['has_detection']:
                    print(f"    - {r['image']}")

    if "data/background" in all_results:
        bg = all_results["data/background"]
        false_positives = sum(1 for r in bg if r['has_detection'])
        print(f"\nBackground (ground truth): {false_positives}/{len(bg)} false positives")
        print(f"  Specificity: {100*(len(bg)-false_positives)/len(bg):.0f}%")
        if false_positives > 0:
            print("  FALSE POSITIVES:")
            for r in bg:
                if r['has_detection']:
                    print(f"    - {r['image']}: {r['species']} ({r['confidence']:.2f})")

    if "data/images" in all_results:
        imgs = all_results["data/images"]
        detected = sum(1 for r in imgs if r['has_detection'])
        print(f"\nAll images: {detected}/{len(imgs)} with detections")

        # Group by species
        species_counts = {}
        for r in imgs:
            if r['has_detection']:
                sp = r['species']
                species_counts[sp] = species_counts.get(sp, 0) + 1

        if species_counts:
            print("\n  Species breakdown:")
            for sp, count in sorted(species_counts.items(), key=lambda x: -x[1]):
                print(f"    {sp}: {count}")

    # Analyze detector confidence distribution
    print(f"\n{'=' * 80}")
    print("DETECTOR CONFIDENCE ANALYSIS")
    print(f"{'=' * 80}")

    if "data/animals" in all_results:
        confs = [r['detector_confidence'] for r in all_results["data/animals"] if r['detector_confidence']]
        if confs:
            print(f"\nAnimals detector confidence: min={min(confs):.3f}, max={max(confs):.3f}, avg={sum(confs)/len(confs):.3f}")

    if "data/background" in all_results:
        confs = [r['detector_confidence'] for r in all_results["data/background"] if r['detector_confidence']]
        if confs:
            print(f"Background detector confidence: min={min(confs):.3f}, max={max(confs):.3f}, avg={sum(confs)/len(confs):.3f}")
        else:
            print("Background detector confidence: no detections (good!)")


if __name__ == "__main__":
    main()
