#!/usr/bin/env python3
"""
Command-line interface for the skin analyzer.

Usage:
    python -m src.cli analyze <image_path> [--output <output_path>] [--verbose]
    python -m src.cli check <image_path>
    python -m src.cli region <image_path> --region <region_name>
"""

import argparse
import json
import sys
from pathlib import Path

from .skin_analyzer import SkinAnalyzer


def analyze_command(args):
    """Run full skin analysis."""
    analyzer = SkinAnalyzer()

    print(f"Analyzing: {args.image}")
    print("-" * 50)

    result = analyzer.analyze(args.image)

    if not result["success"]:
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        return 1

    # Print summary
    summary = result["summary"]
    print("\n=== ANALYSIS SUMMARY ===\n")

    print(f"Face Detected: Yes (confidence: {result['face_detection']['confidence']:.2f})")
    print(f"Skin Coverage: {result['skin_coverage']['percentage']:.1f}%")

    print(f"\nSkin Tone: {summary['skin_tone']}")
    print(f"Undertone: {summary['undertone']}")
    print(f"Texture: {summary['texture']}")
    print(f"Pigmentation: {summary['pigmentation']}")
    print(f"Condition Score: {summary['condition_score']}/100")

    print("\nKey Findings:")
    for finding in summary["key_findings"]:
        print(f"  - {finding}")

    if args.verbose:
        print("\n=== DETAILED RESULTS ===\n")

        # Texture details
        texture = result["texture_analysis"]
        print("Texture Analysis:")
        print(f"  Smoothness Score: {texture['smoothness_score']:.2f}")
        print(f"  Roughness Score: {texture['roughness_score']:.2f}")
        print(f"  Pore Count: {texture['pore_analysis']['pore_count']}")
        print(f"  Pore Density: {texture['pore_analysis']['pore_density']:.2f}")

        glcm = texture["glcm_features"]
        print(f"  GLCM Homogeneity: {glcm['homogeneity']:.3f}")
        print(f"  GLCM Contrast: {glcm['contrast']:.3f}")
        print(f"  GLCM Energy: {glcm['energy']:.3f}")

        # Pigmentation details
        pigmentation = result["pigmentation_analysis"]
        print("\nPigmentation Analysis:")
        print(f"  Dark Spots: {pigmentation['dark_spot_count']}")
        print(f"  Light Spots: {pigmentation['light_spot_count']}")
        print(f"  Evenness Score: {pigmentation['evenness_score']:.2f}")
        print(f"  Redness Score: {pigmentation['redness_areas']['redness_score']:.2f}")

        melanin = pigmentation["melanin_distribution"]
        print(f"  Melanin Index: {melanin['melanin_index']:.2f}")
        print(f"  Melanin Distribution: {melanin['distribution']}")

        # Tone details
        tone = result["tone_analysis"]
        print("\nTone Analysis:")
        print(f"  Dominant Color: {tone['dominant_color']['hex']}")
        print(f"  Color Uniformity: {tone['color_uniformity']:.2f}")
        print(f"  Brightness: {tone['brightness']:.1f}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        analyzer.save_result(result, output_path)
        print(f"\nResults saved to: {output_path}")

    return 0


def check_command(args):
    """Quick check if image is suitable for analysis."""
    analyzer = SkinAnalyzer()

    print(f"Checking: {args.image}")
    print("-" * 50)

    result = analyzer.quick_check(args.image)

    print(f"\nFace Detected: {'Yes' if result['has_face'] else 'No'}")

    if result["has_face"]:
        print(f"Face Count: {result['face_count']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Skin Coverage: {result['skin_percentage']:.1f}%")
        print(f"Suitable for Analysis: {'Yes' if result['is_suitable_for_analysis'] else 'No'}")

        if result["reason"]:
            print(f"Note: {result['reason']}")
    else:
        print("This image is not suitable for skin analysis.")

    return 0 if result.get("is_suitable_for_analysis", False) else 1


def region_command(args):
    """Analyze specific facial region."""
    analyzer = SkinAnalyzer()

    print(f"Analyzing region '{args.region}' in: {args.image}")
    print("-" * 50)

    result = analyzer.analyze_region(args.image, args.region)

    if not result["success"]:
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        return 1

    print(f"\n=== {args.region.upper()} ANALYSIS ===\n")

    # Texture
    texture = result["texture"]
    print("Texture:")
    print(f"  Smoothness: {texture['smoothness_score']:.2f}")
    print(f"  Pore Density: {texture['pore_analysis']['pore_density']:.2f}")

    # Pigmentation
    pigmentation = result["pigmentation"]
    print("\nPigmentation:")
    print(f"  Dark Spots: {pigmentation['dark_spot_count']}")
    print(f"  Evenness: {pigmentation['evenness_score']:.2f}")

    # Tone
    tone = result["tone"]
    print("\nTone:")
    print(f"  Color: {tone['dominant_color']['hex']}")
    print(f"  Uniformity: {tone['color_uniformity']:.2f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Skin Analyzer - Facial skin attribute analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Full skin analysis")
    analyze_parser.add_argument("image", help="Path to image file")
    analyze_parser.add_argument("-o", "--output", help="Output JSON file path")
    analyze_parser.add_argument("-v", "--verbose", action="store_true",
                                help="Show detailed results")

    # Check command
    check_parser = subparsers.add_parser("check", help="Quick suitability check")
    check_parser.add_argument("image", help="Path to image file")

    # Region command
    region_parser = subparsers.add_parser("region", help="Analyze specific region")
    region_parser.add_argument("image", help="Path to image file")
    region_parser.add_argument("-r", "--region", required=True,
                               choices=["forehead", "left_cheek", "right_cheek", "nose", "chin"],
                               help="Facial region to analyze")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "analyze":
        return analyze_command(args)
    elif args.command == "check":
        return check_command(args)
    elif args.command == "region":
        return region_command(args)


if __name__ == "__main__":
    sys.exit(main())
