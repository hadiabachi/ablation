"""
Shared meta-feature category definitions for the property valuation pipeline.
"""

META_FEATURE_CATEGORIES = [
    "Safety & Security",
    "Kitchen & Dining",
    "Comfort Convenience & Connectivity",
    "Parking & Accessibility",
    "Outdoor & Scenery",
    "Family & Community",
    "Entertainment & Leisure",
    "Sports & Fitness",
    "Sustainability Energy Efficiency & Resilience",
    "Luxury & Premium Features",
    "Development & Investment Potential",
    "Legal & Ownership",
    "Healthcare & Hospital Accessibility",
    "Property Orientation & Sunlight Exposure",
    "Cultural Recreational & Heritage Site Proximity",
    "Education & School Proximity",
]

NUMERIC_FEATURES = ["BEDROOM", "BATHROOM", "CARSPACE", "SQM", "LATITUDE", "LONGITUDE"]
OTHER_FEATURES = ["PROPERTY_SUPERCATEGORY"]

TOTAL_FEATURES = NUMERIC_FEATURES + OTHER_FEATURES + META_FEATURE_CATEGORIES
