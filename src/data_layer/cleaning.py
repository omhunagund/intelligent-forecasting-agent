"""
Data cleaning and business-rule normalization for the
Intelligent Business Forecasting Agent.

Responsibilities
----------------
1. Validate the raw table inputs expected by the pipeline.
2. Build the approved revenue-bearing item dataset.
3. Exclude canceled and unavailable orders from the primary
   revenue definition.
4. Preserve overall revenue even when category information is
   missing.
5. Normalize product-category spelling inconsistencies.
6. Apply the approved 15-group business-category hierarchy.
7. Attach the approved 5-region hierarchy.
8. Perform structural and business-rule validation.
9. Return reusable cleaned DataFrames.

This module does NOT:
- modify raw CSV files,
- write processed files automatically,
- engineer lag/rolling/calendar features,
- train forecasting models.

Pipeline orchestration and output persistence will be handled
later by the project orchestration layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd


# ============================================================================
# BUSINESS RULES
# ============================================================================

EXCLUDED_STATUSES: Final[frozenset[str]] = frozenset(
    {"canceled", "unavailable"}
)

MODEL_START: Final[pd.Timestamp] = pd.Timestamp("2017-01-01")
MODEL_END: Final[pd.Timestamp] = pd.Timestamp("2018-08-26")


# ============================================================================
# CATEGORY NORMALIZATION
# ============================================================================

# Raw-category spelling corrections confirmed during EDA.
CATEGORY_ALIASES: Final[dict[str, str]] = {
    "arts_and_craftmanship": "arts_and_craftsmanship",
    "costruction_tools_garden": "construction_tools_garden",
    "costruction_tools_tools": "construction_tools_tools",
    "fashio_female_clothing": "fashion_female_clothing",
    "home_confort": "home_comfort",
}


# Approved 15-group business hierarchy.
CATEGORY_GROUP_MAPPING: Final[dict[str, str]] = {
    # ------------------------------------------------------------------
    # 1. Beauty & Health
    # ------------------------------------------------------------------
    "health_beauty": "Beauty & Health",
    "perfumery": "Beauty & Health",
    "diapers_and_hygiene": "Beauty & Health",

    # ------------------------------------------------------------------
    # 2. Fashion & Accessories
    # ------------------------------------------------------------------
    "watches_gifts": "Fashion & Accessories",
    "fashion_bags_accessories": "Fashion & Accessories",
    "luggage_accessories": "Fashion & Accessories",
    "fashion_shoes": "Fashion & Accessories",
    "fashion_male_clothing": "Fashion & Accessories",
    "fashion_underwear_beach": "Fashion & Accessories",
    "fashion_female_clothing": "Fashion & Accessories",
    "fashion_sport": "Fashion & Accessories",
    "fashion_childrens_clothes": "Fashion & Accessories",

    # ------------------------------------------------------------------
    # 3. Home & Furniture
    # ------------------------------------------------------------------
    "bed_bath_table": "Home & Furniture",
    "furniture_decor": "Home & Furniture",
    "housewares": "Home & Furniture",
    "furniture_living_room": "Home & Furniture",
    "home_comfort": "Home & Furniture",
    "furniture_bedroom": "Home & Furniture",
    "furniture_mattress_and_upholstery": "Home & Furniture",
    "home_comfort_2": "Home & Furniture",

    # ------------------------------------------------------------------
    # 4. Kitchen & Appliances
    # ------------------------------------------------------------------
    "small_appliances": "Kitchen & Appliances",
    "home_appliances_2": "Kitchen & Appliances",
    "home_appliances": "Kitchen & Appliances",
    "air_conditioning": "Kitchen & Appliances",
    "kitchen_dining_laundry_garden_furniture":
        "Kitchen & Appliances",
    "small_appliances_home_oven_and_coffee":
        "Kitchen & Appliances",
    "Kitchen Appliances / Food Preparation Appliances":
        "Kitchen & Appliances",
    "la_cuisine": "Kitchen & Appliances",

    # ------------------------------------------------------------------
    # 5. Electronics & Computing
    # ------------------------------------------------------------------
    "computers_accessories": "Electronics & Computing",
    "computers": "Electronics & Computing",
    "electronics": "Electronics & Computing",
    "PC Gaming": "Electronics & Computing",
    "consoles_games": "Electronics & Computing",
    "audio": "Electronics & Computing",
    "tablets_printing_image": "Electronics & Computing",
    "cine_photo": "Electronics & Computing",

    # ------------------------------------------------------------------
    # 6. Phones & Telecom
    # ------------------------------------------------------------------
    "telephony": "Phones & Telecom",
    "fixed_telephony": "Phones & Telecom",

    # ------------------------------------------------------------------
    # 7. Sports & Leisure
    # ------------------------------------------------------------------
    "sports_leisure": "Sports & Leisure",

    # ------------------------------------------------------------------
    # 8. Kids & Baby
    # ------------------------------------------------------------------
    "toys": "Kids & Baby",
    "baby": "Kids & Baby",

    # ------------------------------------------------------------------
    # 9. Automotive
    # ------------------------------------------------------------------
    "auto": "Automotive",

    # ------------------------------------------------------------------
    # 10. Home Improvement & Garden
    # ------------------------------------------------------------------
    "garden_tools": "Home Improvement & Garden",
    "construction_tools_construction":
        "Home Improvement & Garden",
    "home_construction": "Home Improvement & Garden",
    "construction_tools_lights":
        "Home Improvement & Garden",
    "construction_tools_safety":
        "Home Improvement & Garden",
    "construction_tools_garden":
        "Home Improvement & Garden",
    "construction_tools_tools":
        "Home Improvement & Garden",

    # ------------------------------------------------------------------
    # 11. Office, Business & Services
    # ------------------------------------------------------------------
    "office_furniture": "Office, Business & Services",
    "stationery": "Office, Business & Services",
    "agro_industry_and_commerce":
        "Office, Business & Services",
    "industry_commerce_and_business":
        "Office, Business & Services",
    "signaling_and_security":
        "Office, Business & Services",
    "security_and_services":
        "Office, Business & Services",
    "market_place":
        "Office, Business & Services",

    # ------------------------------------------------------------------
    # 12. Books & Media
    # ------------------------------------------------------------------
    "books_general_interest": "Books & Media",
    "books_technical": "Books & Media",
    "books_imported": "Books & Media",
    "musical_instruments": "Books & Media",
    "music": "Books & Media",
    "dvds_blu_ray": "Books & Media",
    "cds_dvds_musicals": "Books & Media",

    # ------------------------------------------------------------------
    # 13. Food & Beverage
    # ------------------------------------------------------------------
    "food": "Food & Beverage",
    "drinks": "Food & Beverage",
    "food_drink": "Food & Beverage",

    # ------------------------------------------------------------------
    # 14. Pet Supplies
    # ------------------------------------------------------------------
    "pet_shop": "Pet Supplies",

    # ------------------------------------------------------------------
    # 15. Gifts, Arts & Seasonal
    # ------------------------------------------------------------------
    "cool_stuff": "Gifts, Arts & Seasonal",
    "art": "Gifts, Arts & Seasonal",
    "christmas_supplies": "Gifts, Arts & Seasonal",
    "party_supplies": "Gifts, Arts & Seasonal",
    "arts_and_craftsmanship": "Gifts, Arts & Seasonal",
    "flowers": "Gifts, Arts & Seasonal",
}


# Official Olist categories missing from the translation table.
MANUAL_TRANSLATIONS: Final[dict[str, str]] = {
    "pc_gamer": "PC Gaming",
    "portateis_cozinha_e_preparadores_de_alimentos":
        "Kitchen Appliances / Food Preparation Appliances",
}


# ============================================================================
# BRAZILIAN MACRO-REGION MAPPING
# ============================================================================

STATE_TO_REGION: Final[dict[str, str]] = {
    # North
    "AC": "North",
    "AP": "North",
    "AM": "North",
    "PA": "North",
    "RO": "North",
    "RR": "North",
    "TO": "North",

    # Northeast
    "AL": "Northeast",
    "BA": "Northeast",
    "CE": "Northeast",
    "MA": "Northeast",
    "PB": "Northeast",
    "PE": "Northeast",
    "PI": "Northeast",
    "RN": "Northeast",
    "SE": "Northeast",

    # Central-West
    "DF": "Central-West",
    "GO": "Central-West",
    "MT": "Central-West",
    "MS": "Central-West",

    # Southeast
    "ES": "Southeast",
    "MG": "Southeast",
    "RJ": "Southeast",
    "SP": "Southeast",

    # South
    "PR": "South",
    "RS": "South",
    "SC": "South",
}


# ============================================================================
# REQUIRED TABLE SCHEMAS
# ============================================================================

REQUIRED_TABLES: Final[tuple[str, ...]] = (
    "customers",
    "geolocation",
    "order_items",
    "order_payments",
    "order_reviews",
    "orders",
    "products",
    "sellers",
    "category_translation",
)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_input_tables(
    tables: dict[str, pd.DataFrame],
) -> None:
    """
    Validate that the expected raw tables and key columns exist.

    Raises
    ------
    KeyError
        If required tables or columns are missing.
    TypeError
        If a table is not a pandas DataFrame.
    """
    missing_tables = [
        table_name
        for table_name in REQUIRED_TABLES
        if table_name not in tables
    ]

    if missing_tables:
        raise KeyError(
            "Missing required input tables: "
            + ", ".join(missing_tables)
        )

    for table_name in REQUIRED_TABLES:
        if not isinstance(tables[table_name], pd.DataFrame):
            raise TypeError(
                f"Table '{table_name}' must be a pandas DataFrame."
            )

    required_columns: dict[str, set[str]] = {
        "customers": {
            "customer_id",
            "customer_state",
        },
        "orders": {
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        },
        "order_items": {
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        },
        "products": {
            "product_id",
            "product_category_name",
        },
        "category_translation": {
            "product_category_name",
            "product_category_name_english",
        },
    }

    missing_columns: list[str] = []

    for table_name, expected_columns in required_columns.items():
        actual_columns = set(tables[table_name].columns)
        missing = expected_columns - actual_columns

        if missing:
            missing_columns.append(
                f"{table_name}: {sorted(missing)}"
            )

    if missing_columns:
        raise KeyError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {item}" for item in missing_columns
            )
        )


# ============================================================================
# REVENUE-LEVEL CLEANING
# ============================================================================

def build_revenue_items(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the approved item-level revenue dataset.

    Revenue definition:
        item_revenue = price + freight_value

    Excluded order statuses:
        canceled
        unavailable

    Important:
    This function keeps the complete approved revenue universe,
    including rows where product category information is missing.
    Category filtering occurs later in build_category_items().
    """
    order_items = tables["order_items"][
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        ]
    ].copy()

    orders = tables["orders"][
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        ]
    ].copy()

    # Ensure numeric revenue inputs.
    order_items["price"] = pd.to_numeric(
        order_items["price"],
        errors="raise",
    )

    order_items["freight_value"] = pd.to_numeric(
        order_items["freight_value"],
        errors="raise",
    )

    if (order_items["price"] < 0).any():
        raise ValueError("Negative product prices detected.")

    if (order_items["freight_value"] < 0).any():
        raise ValueError("Negative freight values detected.")

    revenue_items = order_items.merge(
        orders,
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    revenue_items = revenue_items[
        ~revenue_items["order_status"].isin(
            EXCLUDED_STATUSES
        )
    ].copy()

    revenue_items["order_purchase_timestamp"] = pd.to_datetime(
        revenue_items["order_purchase_timestamp"],
        errors="raise",
    )

    revenue_items["item_revenue"] = (
        revenue_items["price"]
        + revenue_items["freight_value"]
    )

    return revenue_items


# ============================================================================
# CATEGORY CLEANING
# ============================================================================

def build_category_items(
    revenue_items: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Attach, normalize, and group product categories.

    Rows with missing product-category information are excluded
    from category-level analysis only.

    Overall revenue remains represented in revenue_items.
    """
    products = tables["products"][
        [
            "product_id",
            "product_category_name",
        ]
    ].copy()

    translations = tables["category_translation"][
        [
            "product_category_name",
            "product_category_name_english",
        ]
    ].copy()

    category_items = revenue_items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # Category analysis excludes genuinely missing raw categories.
    category_items = category_items[
        category_items["product_category_name"].notna()
    ].copy()

    category_items = category_items.merge(
        translations,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    # Fill the two categories absent from the translation table.
    category_items["canonical_category"] = (
        category_items["product_category_name_english"]
        .fillna(
            category_items["product_category_name"].map(
                MANUAL_TRANSLATIONS
            )
        )
    )

    unresolved_translation = category_items.loc[
        category_items["canonical_category"].isna(),
        "product_category_name",
    ].drop_duplicates().tolist()

    if unresolved_translation:
        raise ValueError(
            "Unresolved product categories after translation:\n"
            + "\n".join(
                f"  - {value}"
                for value in unresolved_translation
            )
        )

    # Apply approved spelling aliases.
    category_items["canonical_category"] = (
        category_items["canonical_category"]
        .replace(CATEGORY_ALIASES)
    )

    # Every observed canonical category must map to the hierarchy.
    observed_categories = set(
        category_items["canonical_category"]
        .dropna()
        .unique()
    )

    mapping_categories = set(
        CATEGORY_GROUP_MAPPING.keys()
    )

    missing_categories = sorted(
        observed_categories - mapping_categories
    )

    if missing_categories:
        raise ValueError(
            "Observed categories missing from the approved "
            "business-category mapping:\n"
            + "\n".join(
                f"  - {value}"
                for value in missing_categories
            )
        )

    category_items["business_category"] = (
        category_items["canonical_category"]
        .map(CATEGORY_GROUP_MAPPING)
    )

    missing_business_category = int(
        category_items["business_category"].isna().sum()
    )

    if missing_business_category:
        raise ValueError(
            "Rows without a business category: "
            f"{missing_business_category}"
        )

    return category_items


# ============================================================================
# REGION CLEANING
# ============================================================================

def build_region_items(
    revenue_items: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Attach customer state and approved 5-region classification.

    Region assignment is based on customer_state.
    """
    customers = tables["customers"][
        [
            "customer_id",
            "customer_state",
        ]
    ].copy()

    region_items = revenue_items.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    region_items["region"] = (
        region_items["customer_state"]
        .map(STATE_TO_REGION)
    )

    missing_states = sorted(
        region_items.loc[
            region_items["customer_state"].isna(),
            "customer_state",
        ].drop_duplicates()
        .tolist()
    )

    if missing_states:
        raise ValueError(
            "Missing customer-state values encountered."
        )

    unmapped_states = sorted(
        region_items.loc[
            region_items["region"].isna(),
            "customer_state",
        ].drop_duplicates()
        .tolist()
    )

    if unmapped_states:
        raise ValueError(
            "Customer states missing from the approved "
            "5-region mapping:\n"
            + "\n".join(
                f"  - {state}"
                for state in unmapped_states
            )
        )

    return region_items


# ============================================================================
# STRUCTURAL VALIDATION
# ============================================================================

def validate_cleaned_data(
    revenue_items: pd.DataFrame,
    category_items: pd.DataFrame,
    region_items: pd.DataFrame,
) -> dict[str, object]:
    """
    Validate structural integrity of the cleaned datasets.

    Returns
    -------
    dict[str, object]
        Validation metrics suitable for later data-quality reporting.
    """
    duplicate_order_items = int(
        revenue_items[
            ["order_id", "order_item_id"]
        ].duplicated().sum()
    )

    missing_item_revenue = int(
        revenue_items["item_revenue"].isna().sum()
    )

    negative_revenue = int(
        (revenue_items["item_revenue"] < 0).sum()
    )

    missing_category_group = int(
        category_items["business_category"].isna().sum()
    )

    missing_region = int(
        region_items["region"].isna().sum()
    )

    invalid_category_count = (
        category_items["business_category"].nunique()
        != len(set(CATEGORY_GROUP_MAPPING.values()))
    )

    metrics: dict[str, object] = {
        "revenue_rows": len(revenue_items),
        "category_rows": len(category_items),
        "region_rows": len(region_items),
        "duplicate_order_item_keys": duplicate_order_items,
        "missing_item_revenue": missing_item_revenue,
        "negative_item_revenue": negative_revenue,
        "missing_business_category": missing_category_group,
        "missing_region": missing_region,
        "business_category_count":
            category_items["business_category"].nunique(),
        "mapping_category_count":
            len(CATEGORY_GROUP_MAPPING),
        "business_group_count":
            len(set(CATEGORY_GROUP_MAPPING.values())),
        "invalid_business_group_count":
            invalid_category_count,
    }

    if duplicate_order_items:
        raise ValueError(
            "Duplicate (order_id, order_item_id) combinations "
            f"detected: {duplicate_order_items}"
        )

    if missing_item_revenue:
        raise ValueError(
            f"Missing item revenue values: {missing_item_revenue}"
        )

    if negative_revenue:
        raise ValueError(
            f"Negative item revenue values: {negative_revenue}"
        )

    if missing_category_group:
        raise ValueError(
            "Missing business-category assignments: "
            f"{missing_category_group}"
        )

    if missing_region:
        raise ValueError(
            f"Missing region assignments: {missing_region}"
        )

    if invalid_category_count:
        raise ValueError(
            "The observed business-category count does not "
            "match the approved 15-group hierarchy."
        )

    return metrics


# ============================================================================
# MAIN CLEANING FUNCTION
# ============================================================================

def clean_data(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Execute the reusable cleaning pipeline.

    Returns
    -------
    dict[str, pd.DataFrame]
        revenue_items
            Complete approved revenue universe.

        category_items
            Revenue items with canonical and business categories.

        region_items
            Revenue items with customer-state and macro-region data.
    """
    validate_input_tables(tables)

    revenue_items = build_revenue_items(tables)

    category_items = build_category_items(
        revenue_items,
        tables,
    )

    region_items = build_region_items(
        revenue_items,
        tables,
    )

    validate_cleaned_data(
        revenue_items,
        category_items,
        region_items,
    )

    return {
        "revenue_items": revenue_items,
        "category_items": category_items,
        "region_items": region_items,
    }


# ============================================================================
# SUMMARY HELPERS
# ============================================================================

def summarize_cleaned_data(
    cleaned: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Return a compact summary of the cleaned datasets.
    """
    rows = []

    for name, dataframe in cleaned.items():
        row = {
            "dataset": name,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        }

        if "item_revenue" in dataframe.columns:
            row["total_revenue"] = round(
                float(dataframe["item_revenue"].sum()),
                2,
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# STANDALONE SMOKE TEST
# ============================================================================

def main() -> None:
    """
    Standalone smoke test.

    The ingestion module remains responsible for loading raw data.
    This function only demonstrates that the cleaning module can
    consume those tables successfully.
    """
    from src.data_layer.ingestion import load_raw_data

    print("=== DATA CLEANING TEST ===")

    tables = load_raw_data()

    cleaned = clean_data(tables)

    print("\nCleaning completed successfully.")

    print("\n=== CLEANED DATA SUMMARY ===")
    print(
        summarize_cleaned_data(cleaned)
        .to_string(index=False)
    )

    revenue_items = cleaned["revenue_items"]
    category_items = cleaned["category_items"]
    region_items = cleaned["region_items"]

    print("\n=== BUSINESS RULE SUMMARY ===")
    print(
        f"Excluded statuses: "
        f"{sorted(EXCLUDED_STATUSES)}"
    )

    print(
        f"Approved revenue: "
        f"{revenue_items['item_revenue'].sum():,.2f}"
    )

    print(
        f"Category-analysis revenue: "
        f"{category_items['item_revenue'].sum():,.2f}"
    )

    print(
        f"Business categories: "
        f"{category_items['business_category'].nunique()}"
    )

    print(
        f"Regions: "
        f"{region_items['region'].nunique()}"
    )

    print(
        f"Modeling window: "
        f"{MODEL_START.date()} → {MODEL_END.date()}"
    )


if __name__ == "__main__":
    main()