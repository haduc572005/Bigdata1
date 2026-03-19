"""
association.py — Khai phá luật kết hợp: Apriori và FP-Growth.
"""
import logging
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)


class AssociationMiner:
    """
    Khai phá luật kết hợp trên dữ liệu đã rời rạc hóa.

    Attributes
    ----------
    freq_items_  : DataFrame frequent itemsets
    rules_       : DataFrame association rules
    rules_high_  : rules dự đoán Yield cao
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["association"]

    def fit(self, df_disc: pd.DataFrame,
            assoc_cols=None) -> "AssociationMiner":
        if assoc_cols is None:
            assoc_cols = ["Item", "Rain_cat", "Temp_cat", "Pest_cat", "Yield_cat"]

        df_a = df_disc[assoc_cols].dropna()
        transactions = [[str(v) for v in row] for row in df_a.values]

        te = TransactionEncoder()
        te_arr = te.fit_transform(transactions)
        df_onehot = pd.DataFrame(te_arr, columns=te.columns_)
        self.te_ = te
        logger.info(f"Transactions: {len(df_onehot):,}  Items: {len(te.columns_)}")

        min_sup = self.cfg["min_support"]
        max_len = self.cfg["max_len"]
        algo    = self.cfg.get("algorithm", "fpgrowth")

        # Apriori
        logger.info("Chạy Apriori...")
        freq_ap  = apriori(df_onehot, min_support=min_sup,
                           use_colnames=True, max_len=max_len)
        rules_ap = association_rules(freq_ap, metric="lift",
                                     min_threshold=self.cfg["min_lift"])
        logger.info(f"  Apriori → {len(freq_ap)} itemsets, {len(rules_ap)} rules")

        # FP-Growth
        logger.info("Chạy FP-Growth...")
        freq_fp  = fpgrowth(df_onehot, min_support=min_sup,
                            use_colnames=True, max_len=max_len)
        rules_fp = association_rules(freq_fp, metric="lift",
                                     min_threshold=self.cfg["min_lift"])
        logger.info(f"  FP-Growth → {len(freq_fp)} itemsets, {len(rules_fp)} rules")

        self.freq_items_ap_ = freq_ap
        self.rules_ap_      = rules_ap
        self.freq_items_    = freq_fp
        self.rules_         = rules_fp

        # Lọc rules Yield cao
        self.rules_high_ = self._filter_high_yield(rules_fp)
        logger.info(f"  Rules dự đoán Yield cao: {len(self.rules_high_)}")
        return self

    def _filter_high_yield(self, rules: pd.DataFrame) -> pd.DataFrame:
        mask = rules["consequents"].apply(
            lambda x: any(v in str(x) for v in ["Yield_Cao", "Yield_RatCao"])
        )
        return (rules[mask]
                .sort_values(["confidence", "lift"], ascending=False)
                .reset_index(drop=True))

    def top_rules(self, n: int = 10,
                  target_consequent: str = None) -> pd.DataFrame:
        rules = self.rules_high_ if target_consequent else self.rules_
        return (rules.sort_values("lift", ascending=False)
                .head(n)[["antecedents", "consequents",
                           "support", "confidence", "lift"]])

    def compare_algorithms(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Algorithm":    ["Apriori", "FP-Growth"],
            "Freq_Itemsets":[len(self.freq_items_ap_), len(self.freq_items_)],
            "Rules":        [len(self.rules_ap_), len(self.rules_)],
            "Rules_HighYield": [
                len(self._filter_high_yield(self.rules_ap_)),
                len(self.rules_high_)
            ]
        })
