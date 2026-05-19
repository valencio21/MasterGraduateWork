from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

app = FastAPI(
    title="MBA Recommendation API (Advanced Hybrid Engine)",
    description="State-of-the-art Cascade Recommendation System using Multi-Rules (Joint Probability), Additive Synergy, and Strict Filtration."
)

# Словники для зберігання двох типів правил
RULES_DFS = {}
MULTY_RULES_DFS = {}


class CartRequest(BaseModel):
    items: List[str]
    limit: int = 5
    min_conf: float = 0.0
    min_lift: float = 0.0
    min_score: float = 0.0
    exclude_items: Optional[List[str]] = []  # Товари, які заборонено рекомендувати
    exact_match: bool = True


def load_prepared_data(file_path):
    try:
        df = pd.read_parquet(file_path)

        def to_string(x):
            if hasattr(x, '__iter__') and not isinstance(x, str):
                return ', '.join([str(i).strip() for i in x])
            return str(x).strip()

        if 'antecedent_names' in df.columns:
            df['antecedent_str'] = df['antecedent_names'].apply(to_string)
        if 'consequent_names' in df.columns:
            df['consequent_str'] = df['consequent_names'].apply(to_string)

        if 'weighted_score' in df.columns:
            df = df.sort_values(by='weighted_score', ascending=False)

        return df
    except Exception as e:
        print(f"Warning: Could not load {file_path}. Error: {e}")
        return pd.DataFrame()


print("Loading cascade single-rule levels into memory...")
RULES_DFS[3] = load_prepared_data("rules_3_aisle_cluster_department.parquet")
RULES_DFS[2] = load_prepared_data("rules_2_aisle_cluster.parquet")
RULES_DFS[1] = load_prepared_data("rules_1_aisle.parquet")
RULES_DFS[0] = load_prepared_data("rules_0_unfiltered.parquet")

print("Loading cascade multi-rule levels into memory...")
MULTY_RULES_DFS[3] = load_prepared_data("multy_rules_3_aisle_cluster_department.parquet")
MULTY_RULES_DFS[2] = load_prepared_data("multy_rules_2_aisle_cluster.parquet")
MULTY_RULES_DFS[1] = load_prepared_data("multy_rules_1_aisle.parquet")
MULTY_RULES_DFS[0] = load_prepared_data("multy_rules_0_unfiltered.parquet")

print("All levels loaded successfully. Ready to serve!")


@app.get("/")
def read_root():
    return {"message": "Cascade API is running. Go to /docs"}


@app.get("/health")
def health_check():
    """Перевірка стану API та кількості завантажених рівнів (DevOps стандарт)"""
    return {
        "status": "ok",
        "single_rule_levels_loaded": len([k for k, v in RULES_DFS.items() if not v.empty]),
        "multi_rule_levels_loaded": len([k for k, v in MULTY_RULES_DFS.items() if not v.empty])
    }


@app.get("/recommend")
def get_recommendations(
        item: str,
        limit: int = Query(5, description="Maximum number of recommendations"),
        min_conf: float = Query(0.0, description="Minimum Confidence"),
        min_lift: float = Query(0.0, description="Minimum normalized Lift"),
        min_score: float = Query(0.0, description="Minimum Weighted Score"),
        exact_match: bool = Query(True, description="Strict matching to avoid 'Apple' matching 'Pineapple'")
):
    recommendations = []
    seen_consequents = set()

    for level in [3, 2, 1, 0]:
        df = RULES_DFS.get(level)
        if df is None or df.empty or 'antecedent_str' not in df.columns:
            continue

        if exact_match:
            mask = (df['antecedent_str'].str.lower() == item.lower())
        else:
            mask = df['antecedent_str'].str.contains(item, na=False, case=False)

        if min_conf > 0 and 'confidence' in df.columns:
            mask &= (df['confidence'] >= min_conf)
        if min_lift > 0 and 'norm_log_lift' in df.columns:
            mask &= (df['norm_log_lift'] >= min_lift)
        if min_score > 0 and 'weighted_score' in df.columns:
            mask &= (df['weighted_score'] >= min_score)

        level_results = df[mask]

        for _, row in level_results.iterrows():
            consequent = row['consequent_str']

            if consequent not in seen_consequents:
                recommendations.append({
                    "recommended_item": consequent,
                    "filtration_level": level,
                    "confidence": round(row['confidence'], 4) if 'confidence' in row else None,
                    "norm_lift": round(row['norm_log_lift'], 4) if 'norm_log_lift' in row else None,
                    "weighted_score": round(row['weighted_score'], 4) if 'weighted_score' in row else None
                })
                seen_consequents.add(consequent)

            if len(recommendations) >= limit:
                break
        if len(recommendations) >= limit:
            break

    return {
        "target_item": item,
        "filters_applied": {"min_conf": min_conf, "min_lift": min_lift, "min_score": min_score,
                            "exact_match": exact_match},
        "total_returned": len(recommendations),
        "recommendations": recommendations
    }


@app.post("/recommend_cart")
def get_cart_recommendations(payload: CartRequest):
    cart_items_lower = set([item.lower().strip() for item in payload.items])
    exclude_lower = set([item.lower().strip() for item in payload.exclude_items])
    aggregated_recs = {}

    internal_limit_per_item = payload.limit * 3

    for level in [3, 2, 1, 0]:
        df = MULTY_RULES_DFS.get(level)
        if df is None or df.empty or 'antecedent_str' not in df.columns:
            continue

        def is_subset_of_cart(ant_str):
            rule_items = set([x.strip().lower() for x in str(ant_str).split(',')])
            return len(rule_items) > 1 and rule_items.issubset(cart_items_lower)

        mask = df['antecedent_str'].apply(is_subset_of_cart)

        if payload.min_conf > 0 and 'confidence' in df.columns:
            mask &= (df['confidence'] >= payload.min_conf)
        if payload.min_score > 0 and 'weighted_score' in df.columns:
            mask &= (df['weighted_score'] >= payload.min_score)

        for _, row in df[mask].iterrows():
            consequent = row['consequent_str']

            # Пропускаємо, якщо товар вже є в кошику або в чорному списку
            if consequent.lower() in cart_items_lower or consequent.lower() in exclude_lower:
                continue

            # Даємо бонус 1.5x за те, що це справжня спільна синергія кількох товарів
            score = (row['weighted_score'] * 1.5) if 'weighted_score' in row else 0.0

            if consequent not in aggregated_recs:
                aggregated_recs[consequent] = {
                    "recommended_item": consequent,
                    "synergy_score": score,
                    "match_type": "Joint Multi-Rule (High Priority)",
                    "matched_sources": [row['antecedent_str']],
                    "highest_filtration_level": level
                }

    for item in payload.items:
        item_recs_found = 0

        for level in [3, 2, 1, 0]:
            df = RULES_DFS.get(level)
            if df is None or df.empty or 'antecedent_str' not in df.columns:
                continue

            if payload.exact_match:
                mask = (df['antecedent_str'].str.lower() == item.lower())
            else:
                mask = df['antecedent_str'].str.contains(item, na=False, case=False)

            if payload.min_conf > 0 and 'confidence' in df.columns:
                mask &= (df['confidence'] >= payload.min_conf)
            if payload.min_score > 0 and 'weighted_score' in df.columns:
                mask &= (df['weighted_score'] >= payload.min_score)

            level_results = df[mask]

            for _, row in level_results.iterrows():
                consequent = row['consequent_str']

                if consequent.lower() in cart_items_lower or consequent.lower() in exclude_lower:
                    continue

                score = row['weighted_score'] if 'weighted_score' in row else 0.0

                if consequent in aggregated_recs:
                    aggregated_recs[consequent]["synergy_score"] += score
                    if "Additive Logic" not in aggregated_recs[consequent]["match_type"]:
                        aggregated_recs[consequent]["match_type"] += " + Additive Synergy"
                    if item not in aggregated_recs[consequent]["matched_sources"]:
                        aggregated_recs[consequent]["matched_sources"].append(item)
                    aggregated_recs[consequent]["highest_filtration_level"] = max(
                        aggregated_recs[consequent]["highest_filtration_level"], level
                    )
                else:
                    aggregated_recs[consequent] = {
                        "recommended_item": consequent,
                        "synergy_score": score,
                        "match_type": "Additive Logic",
                        "matched_sources": [item],
                        "highest_filtration_level": level
                    }

                item_recs_found += 1
                if item_recs_found >= internal_limit_per_item:
                    break

            if item_recs_found >= internal_limit_per_item:
                break

    # Сортування та форматування результатів
    sorted_recs = sorted(aggregated_recs.values(), key=lambda x: x["synergy_score"], reverse=True)
    final_recs = sorted_recs[:payload.limit]

    for rec in final_recs:
        rec["synergy_score"] = round(rec["synergy_score"], 4)

    return {
        "cart_items": payload.items,
        "excluded_items": payload.exclude_items,
        "filters_applied": {
            "min_score": payload.min_score,
            "exact_match": payload.exact_match
        },
        "total_returned": len(final_recs),
        "recommendations": final_recs
    }


@app.get("/promotions")
def get_promotions(
        target_item: str = Query(..., description="Consequent (Item to sell)"),
        limit: int = Query(5, description="Maximum number of items"),
        min_conf: float = Query(0.0, description="Minimum Confidence"),
        min_lift: float = Query(0.0, description="Minimum normalized Lift"),
        min_score: float = Query(0.0, description="Minimum Weighted Score"),
        exact_match: bool = Query(True, description="Strict matching")
):
    promotions = []
    seen_antecedents = set()

    for level in [3, 2, 1, 0]:
        df = RULES_DFS.get(level)
        if df is None or df.empty or 'consequent_str' not in df.columns:
            continue

        if exact_match:
            mask = (df['consequent_str'].str.lower() == target_item.lower())
        else:
            mask = df['consequent_str'].str.contains(target_item, na=False, case=False)

        if min_conf > 0 and 'confidence' in df.columns:
            mask &= (df['confidence'] >= min_conf)
        if min_lift > 0 and 'norm_log_lift' in df.columns:
            mask &= (df['norm_log_lift'] >= min_lift)
        if min_score > 0 and 'weighted_score' in df.columns:
            mask &= (df['weighted_score'] >= min_score)

        level_results = df[mask]

        for _, row in level_results.iterrows():
            antecedent = row['antecedent_str']

            if antecedent not in seen_antecedents:
                promotions.append({
                    "item_to_promote": antecedent,
                    "filtration_level": level,
                    "confidence": round(row['confidence'], 4) if 'confidence' in row else None,
                    "norm_lift": round(row['norm_log_lift'], 4) if 'norm_log_lift' in row else None,
                    "weighted_score": round(row['weighted_score'], 4) if 'weighted_score' in row else None
                })
                seen_antecedents.add(antecedent)

            if len(promotions) >= limit:
                break
        if len(promotions) >= limit:
            break

    return {
        "target_item_to_sell": target_item,
        "filters_applied": {"min_conf": min_conf, "min_lift": min_lift, "min_score": min_score,
                            "exact_match": exact_match},
        "total_returned": len(promotions),
        "promotions": promotions
    }