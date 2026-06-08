import pandas as pd
import numpy as np
from collections import defaultdict

class WangMendelExtractor:
    """
    Implements the Wang-Mendel algorithm for extracting fuzzy rules from numerical data.
    Steps:
    1. Divide the input and output spaces into fuzzy regions (uniform partition).
    2. Generate fuzzy rules from given data pairs.
    3. Assign a degree to each rule.
    4. Create a combined fuzzy rule base (resolve conflicts).
    """
    def __init__(self, n_terms=5, mf_type='triangular'):
        self.n_terms = n_terms
        self.mf_type = mf_type
        self.variables = {}  
        self.rules = []      

    def partition_space(self, df):
        self.variables = {}
        for col in df.columns:
            vmin, vmax = df[col].min(), df[col].max()
            if vmin == vmax:
                vmax = vmin + 1.0
            
            centers = np.linspace(vmin, vmax, self.n_terms)
            width = (vmax - vmin) / (self.n_terms - 1)
            
            terms = {}
            for i, c in enumerate(centers):
                term_name = f"T{i+1}"
                if self.mf_type == 'triangular':
                    left = c - width if i > 0 else vmin - width
                    right = c + width if i < self.n_terms - 1 else vmax + width
                    terms[term_name] = {"type": "triangular", "params": [left, c, right], "center": c}
            
            self.variables[col] = {
                "min": float(vmin),
                "max": float(vmax),
                "terms": terms
            }

    def evaluate_mfs(self, var_name, value):
        memberships = {}
        var_data = self.variables[var_name]
        for term_name, term_data in var_data["terms"].items():
            params = term_data["params"]
            left, center, right = params
            if value <= left or value >= right:
                mu = 0.0
            elif value == center:
                mu = 1.0
            elif value < center:
                mu = (value - left) / (center - left)
            else:
                mu = (right - value) / (right - center)
            memberships[term_name] = mu
        return memberships

    def fit(self, df, target_col):
        self.partition_space(df)
        input_cols = [c for c in df.columns if c != target_col]
        candidate_rules = defaultdict(list)
        
        for idx, row in df.iterrows():
            rule_antecedent = {}
            ant_degree = 1.0
            
            for col in input_cols:
                mus = self.evaluate_mfs(col, row[col])
                best_term = max(mus.items(), key=lambda x: x[1])
                rule_antecedent[col] = best_term[0]
                ant_degree *= best_term[1]
            
            mus_out = self.evaluate_mfs(target_col, row[target_col])
            best_out_term = max(mus_out.items(), key=lambda x: x[1])
            rule_consequent = {target_col: best_out_term[0]}
            rule_degree = ant_degree * best_out_term[1]
            
            ant_sig = frozenset(rule_antecedent.items())
            candidate_rules[ant_sig].append({
                "antecedents": rule_antecedent,
                "consequent": rule_consequent,
                "degree": rule_degree
            })
            
        self.rules = []
        for ant_sig, rules_group in candidate_rules.items():
            best_rule = max(rules_group, key=lambda x: x["degree"])
            self.rules.append(best_rule)

    def export_config(self, target_col):
        config = {
            "variables": {},
            "rules": []
        }
        for var_name, var_data in self.variables.items():
            role = "output" if var_name == target_col else "input"
            terms_config = {}
            for t_name, t_data in var_data["terms"].items():
                terms_config[t_name] = {
                    "type": "triangular",
                    "params": t_data["params"]
                }
            config["variables"][var_name] = {
                "role": role,
                "min_val": var_data["min"],
                "max_val": var_data["max"],
                "terms": terms_config
            }
        for rule in self.rules:
            config["rules"].append({
                "antecedents": rule["antecedents"],
                "consequent": rule["consequent"],
                "weight": 1.0
            })
        return config
