import base64
import binascii
import io
import math
import os
import secrets
import time
from typing import Annotated, Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, Field

from fuzzy_torch import FuzzyModule, FuzzyVariable, FuzzyRule, TriangularMF, TrapezoidalMF, GaussianMF, BellMF, SigmoidMF
from fuzzy_torch.defuzz import Centroid
from eautomfis.wang_mendel import WangMendelExtractor

# --- Settings (environment) ---
# Comma-separated origins allowed to call the API from another site. The bundled UI is
# served from the same origin, so CORS is off unless this is set.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("FIS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
SESSION_TTL = int(os.environ.get("FIS_SESSION_TTL", 4 * 3600))  # seconds of inactivity
MAX_SESSIONS = int(os.environ.get("FIS_MAX_SESSIONS", 1000))
MAX_CSV_BYTES = int(os.environ.get("FIS_MAX_CSV_BYTES", 5 * 1024 * 1024))
SESSION_COOKIE = "fis_session"

app = FastAPI(title="Fuzzy Inference System API")

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Membership functions ---
# short name -> (class, parameter attribute names)
MF_SPECS = {
    "tri": (TriangularMF, ["a", "b", "c"]),
    "trap": (TrapezoidalMF, ["a", "b", "c", "d"]),
    "gauss": (GaussianMF, ["mu", "sigma"]),
    "bell": (BellMF, ["a", "b", "c"]),
    "sig": (SigmoidMF, ["b", "c"]),
}
MF_ALIASES = {
    "tri": "tri", "triangular": "tri", "triangularmf": "tri", "trimf": "tri",
    "trap": "trap", "trapezoidal": "trap", "trapezoidalmf": "trap", "trapmf": "trap",
    "gauss": "gauss", "gaussian": "gauss", "gaussianmf": "gauss", "gaussmf": "gauss",
    "bell": "bell", "bellmf": "bell", "gbellmf": "bell",
    "sig": "sig", "sigmoid": "sig", "sigmoidmf": "sig", "sigmf": "sig",
}


def mf_kind(type_name: str) -> str:
    kind = MF_ALIASES.get(str(type_name).strip().lower())
    if kind is None:
        raise ValueError(f"Unknown membership function type '{type_name}'")
    return kind


def make_mf(type_name: str, params: List[float]):
    """Validates parameters and builds a membership function."""
    kind = mf_kind(type_name)
    cls, names = MF_SPECS[kind]
    if len(params) != len(names):
        raise ValueError(f"{cls.__name__} needs {len(names)} parameters ({', '.join(names)}), got {len(params)}")
    p = [float(v) for v in params]
    if not all(math.isfinite(v) for v in p):
        raise ValueError("Parameters must be finite numbers")
    if kind in ("tri", "trap") and p != sorted(p):
        raise ValueError(f"{cls.__name__} parameters must be in ascending order ({' <= '.join(names)})")
    if kind == "gauss" and p[1] <= 0:
        raise ValueError("Gaussian sigma must be greater than 0")
    if kind == "bell" and (p[0] == 0 or p[1] <= 0):
        raise ValueError("Bell 'a' must be non-zero and 'b' greater than 0")
    return cls(*p)


def mf_to_dict(mf) -> Dict[str, Any]:
    for cls, names in MF_SPECS.values():
        if type(mf) is cls:
            return {"type": cls.__name__, "params": [getattr(mf, n).item() for n in names]}
    raise ValueError(f"Unsupported membership function {type(mf).__name__}")


def check_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Name must not be empty")
    if "/" in value:
        raise ValueError("Name must not contain '/'")
    if len(value) > 64:
        raise ValueError("Name must be at most 64 characters")
    return value


def http_400(e: Exception):
    msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
    return HTTPException(status_code=400, detail=msg)


# --- Per-session state (Builder Pattern) ---
class FuzzyBuilder:
    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.roles: Dict[str, str] = {}  # var_name -> 'input' | 'output'
        self.rules_data: List[Dict] = []  # raw rule config, in system rule order
        self._system: Optional[FuzzyModule] = None

    def invalidate(self):
        """Call after any change; the system is rebuilt on next use."""
        self._system = None

    def reset(self):
        self.variables, self.roles, self.rules_data = {}, {}, []
        self.invalidate()

    def get_variable(self, name: str) -> FuzzyVariable:
        if name not in self.variables:
            raise HTTPException(status_code=404, detail=f"Variable '{name}' not found")
        return self.variables[name]

    def add_variable(self, name: str, min_val: float, max_val: float, role: str):
        name = check_name(name)
        if role not in ("input", "output"):
            raise ValueError("Role must be 'input' or 'output'")
        if not (math.isfinite(min_val) and math.isfinite(max_val)) or min_val >= max_val:
            raise ValueError(f"Variable '{name}': min must be smaller than max")
        if name in self.variables:
            # Redefining a variable keeps its terms but changes range/role
            old = self.variables[name]
            var = FuzzyVariable(name, min_val, max_val)
            for t_name, mf in old.terms.items():
                var.add_term(t_name, mf)
            if role != self.roles.get(name):
                self.rules_data = [r for r in self.rules_data if name not in r["antecedents"] and name not in r["consequent"]]
        else:
            var = FuzzyVariable(name, min_val, max_val)
        self.variables[name] = var
        self.roles[name] = role
        self.invalidate()

    def add_term(self, var_name: str, term_name: str, type_name: str, params: List[float]):
        var = self.get_variable(var_name)
        var.add_term(check_name(term_name), make_mf(type_name, params))
        self.invalidate()

    def add_rule(self, antecedents: Dict[str, str], consequent: Dict[str, str]):
        if not antecedents:
            raise ValueError("A rule needs at least one antecedent (IF part)")
        if len(consequent) != 1:
            raise ValueError("A rule needs exactly one consequent (THEN part)")
        for v, t in antecedents.items():
            if v not in self.variables or t not in self.variables[v].terms:
                raise ValueError(f"Invalid antecedent {v} IS {t}")
            if self.roles.get(v) != "input":
                raise ValueError(f"Antecedent '{v}' is not an input variable")
        (c_var, c_term), = consequent.items()
        if c_var not in self.variables or c_term not in self.variables[c_var].terms:
            raise ValueError(f"Invalid consequent {c_var} IS {c_term}")
        if self.roles.get(c_var) != "output":
            raise ValueError(f"Consequent '{c_var}' is not an output variable")
        self.rules_data.append({"antecedents": dict(antecedents), "consequent": dict(consequent)})
        self.invalidate()

    def delete_variable(self, name: str):
        self.get_variable(name)
        del self.variables[name]
        self.roles.pop(name, None)
        # Remove rules referencing this variable
        self.rules_data = [
            r for r in self.rules_data
            if name not in r['antecedents'] and name not in r['consequent']
        ]
        self.invalidate()

    def delete_term(self, var_name: str, term_name: str):
        var = self.get_variable(var_name)
        if term_name not in var.terms:
            raise HTTPException(status_code=404, detail=f"Term '{term_name}' not found in '{var_name}'")
        del var.terms[term_name]
        # Remove rules referencing this term
        self.rules_data = [
            r for r in self.rules_data
            if r['antecedents'].get(var_name) != term_name and r['consequent'].get(var_name) != term_name
        ]
        self.invalidate()

    def delete_rule(self, index: int):
        if not 0 <= index < len(self.rules_data):
            raise HTTPException(status_code=404, detail=f"Rule {index} not found")
        self.rules_data.pop(index)
        self.invalidate()

    def load_config(self, config: Dict[str, Any]):
        """Replaces the state with `config`. On error the current state is left untouched.

        Accepts the /export format as well as older/alternative spellings
        (min_val/max_val, terms as a dict, 'triangular' type names).
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object")
        staged = FuzzyBuilder()
        variables = config.get("variables", {})
        if not isinstance(variables, dict):
            raise ValueError("'variables' must be an object")
        for name, data in variables.items():
            lo, hi = data.get("min", data.get("min_val")), data.get("max", data.get("max_val"))
            if lo is None or hi is None:
                raise ValueError(f"Variable '{name}' is missing min/max")
            staged.add_variable(name, float(lo), float(hi), data.get("role", "input"))
            terms = data.get("terms", [])
            if isinstance(terms, dict):
                terms = [{"name": k, **v} for k, v in terms.items()]
            for term in terms:
                try:
                    staged.add_term(name, term["name"], term["type"], term["params"])
                except (KeyError, TypeError) as e:
                    raise ValueError(f"Variable '{name}': malformed term {term!r}") from e
                except ValueError as e:
                    raise ValueError(f"Variable '{name}', term '{term.get('name')}': {e}") from e
        for i, rule in enumerate(config.get("rules", [])):
            try:
                staged.add_rule(rule.get("antecedents", {}), rule.get("consequent", {}))
            except (AttributeError, ValueError) as e:
                raise ValueError(f"Rule {i}: {e}") from e
        self.variables, self.roles, self.rules_data = staged.variables, staged.roles, staged.rules_data
        self.invalidate()

    def export_config(self, terms_as_dict: bool = False) -> Dict[str, Any]:
        vars_out = {}
        for name, var in self.variables.items():
            if terms_as_dict:
                terms = {t: mf_to_dict(mf) for t, mf in var.terms.items()}
            else:
                terms = [{"name": t, **mf_to_dict(mf)} for t, mf in var.terms.items()]
            vars_out[name] = {"min": var.range_min, "max": var.range_max, "role": self.roles.get(name, "input"), "terms": terms}
        return {"variables": vars_out, "rules": self.rules_data}

    def output_name(self) -> str:
        outputs = [n for n, r in self.roles.items() if r == "output"]
        if not outputs:
            raise ValueError("No output variable defined. Create an output variable in the Editor tab.")
        if len(outputs) > 1:
            raise ValueError(f"Only one output variable is supported, found {len(outputs)}: {', '.join(outputs)}")
        return outputs[0]

    def required_inputs(self) -> List[str]:
        names = []
        for r in self.rules_data:
            for v in r["antecedents"]:
                if v not in names:
                    names.append(v)
        return names

    def system(self) -> FuzzyModule:
        """Returns the built system, (re)building it if the configuration changed."""
        if self._system is not None:
            return self._system
        try:
            out_name = self.output_name()
            output = self.variables[out_name]
            if not output.terms:
                raise ValueError(f"Output variable '{out_name}' has no fuzzy sets yet.")
            if not self.rules_data:
                raise ValueError("No rules defined yet. Add at least one rule in the Rules tab.")
            rules = []
            for r in self.rules_data:
                (c_name, c_term), = r["consequent"].items()
                antecedents = [(self.variables[v], t) for v, t in r["antecedents"].items()]
                rules.append(FuzzyRule(antecedent=antecedents, consequent=(self.variables[c_name], c_term)))
            inputs = [v for k, v in self.variables.items() if self.roles.get(k) == "input"]
            self._system = FuzzyModule(inputs=inputs, output=output, rules=rules,
                                       defuzz_method=Centroid(output.range_min, output.range_max))
        except (ValueError, KeyError) as e:
            raise http_400(e)
        return self._system


class SessionStore:
    """In-memory builders keyed by a browser cookie, so visitors don't share a system."""

    def __init__(self):
        self._sessions: Dict[str, List] = {}  # sid -> [builder, last_seen]

    def get(self, sid: Optional[str]) -> Tuple[str, FuzzyBuilder]:
        now = time.monotonic()
        entry = self._sessions.get(sid) if sid else None
        if entry is None:
            self._evict(now)
            if not sid or len(sid) > 64:
                sid = secrets.token_urlsafe(24)
            entry = self._sessions[sid] = [FuzzyBuilder(), now]
        entry[1] = now
        return sid, entry[0]

    def _evict(self, now: float):
        for sid in [s for s, (_, seen) in self._sessions.items() if now - seen > SESSION_TTL]:
            del self._sessions[sid]
        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda s: self._sessions[s][1])
            del self._sessions[oldest]


sessions = SessionStore()


@app.middleware("http")
async def session_middleware(request: Request, call_next):
    cookie = request.cookies.get(SESSION_COOKIE)
    sid, request.state.builder = sessions.get(cookie)
    response = await call_next(request)
    if sid != cookie:
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax",
                            secure=request.url.scheme == "https")
    return response


def get_builder(request: Request) -> FuzzyBuilder:
    return request.state.builder


# --- Pydantic Models ---
Name = Annotated[str, AfterValidator(check_name)]


class CreateVariable(BaseModel):
    name: Name
    min_val: float
    max_val: float
    role: str = "input"  # input or output


class CreateMF(BaseModel):
    name: Name
    type: str  # tri, trap, gauss, bell, sig
    params: List[float]


class CreateRule(BaseModel):
    antecedents: Dict[str, str]  # {VarName: TermName}
    consequent: Dict[str, str]  # {VarName: TermName} (single consequent)


class SimulateRequest(BaseModel):
    inputs: Dict[str, float]


class WangMendelRequest(BaseModel):
    csv_data: str  # base64 encoded csv
    target_col: str
    n_terms: int = Field(5, ge=2, le=15)


# --- Endpoints ---
# State endpoints are `async def` so they run one at a time on the event loop and
# never race on a session's builder (e.g. a simulation poll during an edit).

@app.post("/wang-mendel")
def run_wang_mendel(req: WangMendelRequest):
    if len(req.csv_data) > MAX_CSV_BYTES * 4 // 3 + 4:
        raise HTTPException(status_code=413, detail=f"CSV is too large (limit {MAX_CSV_BYTES // (1024 * 1024)} MB)")
    try:
        csv_str = base64.b64decode(req.csv_data, validate=True).decode('utf-8-sig')
        df = pd.read_csv(io.StringIO(csv_str))
    except (binascii.Error, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_col}' not found in data.")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise HTTPException(status_code=400, detail=f"All columns must be numeric; not numeric: {', '.join(non_numeric)}")
    df = df.dropna()
    if df.shape[1] < 2 or df.empty:
        raise HTTPException(status_code=400, detail="Need at least one input column, the target column and one complete row.")
    bad = [c for c in df.columns if "/" in c or len(c) > 64]
    if bad:
        raise HTTPException(status_code=400, detail=f"Column names must not contain '/' or exceed 64 characters: {', '.join(bad)}")

    extractor = WangMendelExtractor(n_terms=req.n_terms)
    extractor.fit(df, req.target_col)
    return extractor.export_config(req.target_col)


@app.get("/config")
async def get_config(b: FuzzyBuilder = Depends(get_builder)):
    """Returns current system configuration (terms keyed by name)."""
    return b.export_config(terms_as_dict=True)


@app.get("/export")
async def export_system(b: FuzzyBuilder = Depends(get_builder)):
    """Export the full system configuration, in the format /import accepts."""
    return b.export_config()


@app.post("/import")
async def import_system(config: Dict[str, Any], b: FuzzyBuilder = Depends(get_builder)):
    """Restores system from config."""
    try:
        b.load_config(config)
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {e}")
    return {"message": "System imported successfully"}


@app.post("/reset")
async def reset_system(b: FuzzyBuilder = Depends(get_builder)):
    b.reset()
    return {"message": "System reset"}


def _tri_var(lo, hi, role, terms):
    return {"min": lo, "max": hi, "role": role,
            "terms": [{"name": n, "type": "tri", "params": p} for n, p in terms.items()]}


def _rule(cons, **ants):
    return {"antecedents": ants, "consequent": cons}


PRESETS = {
    # Exercise: students define the terms and rules themselves
    "dc_motor": {
        "variables": {
            "error": _tri_var(-100, 100, "input", {}),
            "error_dot": _tri_var(-50, 50, "input", {}),
            "voltage": _tri_var(-12, 12, "output", {}),
        },
        "rules": [],
    },
    "tank_broken": {
        "variables": {
            "level": _tri_var(-1, 1, "input", {"low": [-1, -1, 0], "okay": [-0.5, 0, 0.5], "high": [0, 1, 1]}),  # Deliberately broken (too wide)
            "rate": _tri_var(-0.1, 0.1, "input", {"negative": [-0.1, -0.1, 0], "none": [-0.01, 0, 0.01], "positive": [0, 0.1, 0.1]}),
            "valve": _tri_var(-1, 1, "output", {"close_fast": [-1, -1, -0.3], "close_slow": [-0.6, -0.3, 0], "no_change": [-0.1, 0, 0.1],
                                                "open_slow": [0, 0.2, 0.5], "open_fast": [0.3, 1, 1]}),
        },
        # Broken rules: swapped consequents and missing rule
        "rules": [
            _rule({"valve": "no_change"}, level="okay"),
            _rule({"valve": "close_fast"}, level="low"),
            _rule({"valve": "open_fast"}, level="high"),
            _rule({"valve": "close_slow"}, level="okay", rate="positive"),
        ],
    },
    # MathWorks sltank: level error (current - setpoint) and its rate, valve rate of change
    "tank": {
        "variables": {
            "level": _tri_var(-1, 1, "input", {"low": [-1, -1, 0], "okay": [-0.1, 0, 0.1], "high": [0, 1, 1]}),
            "rate": _tri_var(-0.1, 0.1, "input", {"negative": [-0.1, -0.1, 0], "none": [-0.01, 0, 0.01], "positive": [0, 0.1, 0.1]}),
            "valve": _tri_var(-1, 1, "output", {"close_fast": [-1, -1, -0.3], "close_slow": [-0.6, -0.3, 0], "no_change": [-0.1, 0, 0.1],
                                                "open_slow": [0, 0.2, 0.5], "open_fast": [0.3, 1, 1]}),
        },
        "rules": [
            _rule({"valve": "no_change"}, level="okay"),
            _rule({"valve": "open_fast"}, level="low"),
            _rule({"valve": "close_fast"}, level="high"),
            _rule({"valve": "close_slow"}, level="okay", rate="positive"),
            _rule({"valve": "open_slow"}, level="okay", rate="negative"),
        ],
    },
    "crane": {
        "variables": {
            "PosErr": _tri_var(-10, 10, "input", {"Neg": [-10, -10, 0], "Zero": [-1, 0, 1], "Pos": [0, 10, 10]}),
            "Vel": _tri_var(-5, 5, "input", {"Neg": [-5, -5, 0], "Zero": [-0.5, 0, 0.5], "Pos": [0, 5, 5]}),
            "Angle": _tri_var(-1, 1, "input", {"Neg": [-1, -1, 0], "Zero": [-0.1, 0, 0.1], "Pos": [0, 1, 1]}),
            "W": _tri_var(-2, 2, "input", {"Neg": [-2, -2, 0], "Zero": [-0.2, 0, 0.2], "Pos": [0, 2, 2]}),
            "Force": _tri_var(-30, 30, "output", {"Left": [-30, -30, 0], "Zero": [-2, 0, 2], "Right": [0, 30, 30]}),
        },
        # Tuned Rules. Anti-sway: move the trolley toward the side the load swings to
        "rules": [
            _rule({"Force": "Right"}, PosErr="Pos"),
            _rule({"Force": "Left"}, PosErr="Neg"),
            _rule({"Force": "Left"}, Vel="Pos"),
            _rule({"Force": "Right"}, Vel="Neg"),
            _rule({"Force": "Right"}, Angle="Pos"),
            _rule({"Force": "Left"}, Angle="Neg"),
            _rule({"Force": "Right"}, W="Pos"),
            _rule({"Force": "Left"}, W="Neg"),
        ],
    },
    # Distance error and its rate of change
    "robot_linear": {
        "variables": {
            "DistErr": _tri_var(-10, 10, "input", {"Neg": [-10, -10, 0], "Zero": [-1, 0, 1], "Pos": [0, 10, 10]}),
            "Rate": _tri_var(-2, 2, "input", {"Neg": [-2, -2, 0], "Zero": [-0.3, 0, 0.3], "Pos": [0, 2, 2]}),
            "Power": _tri_var(-1, 1, "output", {"Reverse": [-1, -1, 0], "Stop": [-0.2, 0, 0.2], "Forward": [0, 1, 1]}),
        },
        "rules": [
            _rule({"Power": "Forward"}, DistErr="Pos"),
            _rule({"Power": "Reverse"}, DistErr="Neg"),
            _rule({"Power": "Stop"}, DistErr="Zero"),
            _rule({"Power": "Reverse"}, DistErr="Zero", Rate="Pos"),
            _rule({"Power": "Forward"}, DistErr="Zero", Rate="Neg"),
        ],
    },
    # Angular velocity error and its rate
    "robot_angular": {
        "variables": {
            "AngErr": _tri_var(-5, 5, "input", {"Neg": [-5, -5, 0], "Zero": [-0.5, 0, 0.5], "Pos": [0, 5, 5]}),
            "AngRate": _tri_var(-2, 2, "input", {"Neg": [-2, -2, 0], "Zero": [-0.3, 0, 0.3], "Pos": [0, 2, 2]}),
            "Steering": _tri_var(-1, 1, "output", {"Left": [-1, -1, 0], "None": [-0.2, 0, 0.2], "Right": [0, 1, 1]}),
        },
        "rules": [
            _rule({"Steering": "Right"}, AngErr="Pos"),
            _rule({"Steering": "Left"}, AngErr="Neg"),
            _rule({"Steering": "None"}, AngErr="Zero"),
            _rule({"Steering": "Left"}, AngErr="Zero", AngRate="Pos"),
            _rule({"Steering": "Right"}, AngErr="Zero", AngRate="Neg"),
        ],
    },
    # 4 inputs: distance error, dDist, heading error, dHeading; output: linear velocity [0, 1]
    "robot_nav": {
        "variables": {
            "DistErr": _tri_var(0, 10, "input", {"Close": [0, 0, 2], "Medium": [1, 5, 9], "Far": [7, 10, 10]}),
            "dDist": _tri_var(-2, 2, "input", {"Neg": [-2, -2, 0], "Zero": [-0.3, 0, 0.3], "Pos": [0, 2, 2]}),
            "HeadErr": _tri_var(-3.15, 3.15, "input", {"Neg": [-3.15, -3.15, 0], "Zero": [-0.3, 0, 0.3], "Pos": [0, 3.15, 3.15]}),
            "dHead": _tri_var(-1, 1, "input", {"Neg": [-1, -1, 0], "Zero": [-0.15, 0, 0.15], "Pos": [0, 1, 1]}),
            "Velocity": _tri_var(0, 1, "output", {"Stop": [0, 0, 0.2], "Slow": [0.1, 0.3, 0.5], "Fast": [0.4, 1, 1]}),
        },
        "rules": [
            _rule({"Velocity": "Fast"}, DistErr="Far", HeadErr="Zero"),
            _rule({"Velocity": "Slow"}, DistErr="Medium"),
            _rule({"Velocity": "Stop"}, DistErr="Close"),
            _rule({"Velocity": "Slow"}, HeadErr="Pos"),
            _rule({"Velocity": "Slow"}, HeadErr="Neg"),
        ],
    },
}


@app.post("/preset/{scenario_id}")
async def load_preset(scenario_id: str, b: FuzzyBuilder = Depends(get_builder)):
    """Loads a FIS preset for a specific laboratory scenario."""
    if scenario_id not in PRESETS:
        raise HTTPException(status_code=404, detail="Scenario not found")
    b.load_config(PRESETS[scenario_id])
    return {"message": f"Preset for {scenario_id} loaded"}


@app.post("/variable")
async def create_variable(data: CreateVariable, b: FuzzyBuilder = Depends(get_builder)):
    try:
        b.add_variable(data.name, data.min_val, data.max_val, data.role)
    except ValueError as e:
        raise http_400(e)
    return {"message": f"Variable {data.name} created"}


@app.delete("/variable/{name}")
async def delete_variable(name: str, b: FuzzyBuilder = Depends(get_builder)):
    b.delete_variable(name)
    return {"message": f"Variable {name} deleted"}


@app.delete("/variable/{name}/term/{term}")
async def delete_term(name: str, term: str, b: FuzzyBuilder = Depends(get_builder)):
    b.delete_term(name, term)
    return {"message": f"Term {term} deleted from {name}"}


@app.delete("/rule/{index}")
async def delete_rule(index: int, b: FuzzyBuilder = Depends(get_builder)):
    b.delete_rule(index)
    return {"message": f"Rule {index} deleted"}


@app.post("/variable/{name}/term")
async def add_term(name: str, data: CreateMF, b: FuzzyBuilder = Depends(get_builder)):
    try:
        b.add_term(name, data.name, data.type, data.params)
    except ValueError as e:
        raise http_400(e)
    return {"message": f"Term {data.name} added to {name}"}


@app.post("/rule")
async def add_rule(data: CreateRule, b: FuzzyBuilder = Depends(get_builder)):
    try:
        b.add_rule(data.antecedents, data.consequent)
    except ValueError as e:
        raise http_400(e)
    return {"message": "Rule added"}


@app.post("/build")
async def build_system(b: FuzzyBuilder = Depends(get_builder)):
    """Checks that the system can be built. /simulate also builds on demand."""
    b.invalidate()
    b.system()
    return {"message": "System built successfully"}


@app.post("/simulate")
async def simulate(data: SimulateRequest, b: FuzzyBuilder = Depends(get_builder)):
    system = b.system()
    missing = [v for v in b.required_inputs() if v not in data.inputs]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing input value(s): {', '.join(missing)}")
    if not all(math.isfinite(v) for v in data.inputs.values()):
        raise HTTPException(status_code=400, detail="Input values must be finite numbers")

    inputs_torch = {k: torch.tensor([v]) for k, v in data.inputs.items() if k in system.input_variables}
    with torch.no_grad():
        activations = system.get_rule_activations(inputs_torch)[0]  # (num_rules,)
        output = system(inputs_torch).item()  # NaN when no rule fires

        input_memberships = {}
        for var_name, val in data.inputs.items():
            if b.roles.get(var_name) == "input" and var_name in b.variables:
                var = b.variables[var_name]
                input_memberships[var_name] = {t: mf(torch.tensor([val])).item() for t, mf in var.terms.items()}

        universe = system.defuzz.universe
        aggregated_mf = torch.zeros_like(universe)
        rule_contributions = []
        for i, rule in enumerate(system.rules):
            strength = activations[i].item()
            if strength <= 1e-4:
                continue
            cons_mf_values = system.output_variable.terms[rule.consequent[1]](universe)
            rule_output = system.implication(torch.tensor([strength]), cons_mf_values)
            aggregated_mf = system.aggregation(aggregated_mf, rule_output)

            rd = b.rules_data[i]
            (c_var, c_term), = rd["consequent"].items()
            parts = [f"{v} IS {t}" for v, t in rd["antecedents"].items()]
            rule_contributions.append({
                "index": i,
                "description": f"IF {' AND '.join(parts)} THEN {c_var} IS {c_term}",
                "config": rd,
                "strength": strength,
                "consequent_term": c_term,
                "clipped_shape": rule_output.tolist(),
            })

    no_rule_fired = not math.isfinite(output)
    return {
        "output": None if no_rule_fired else output,
        "no_rule_fired": no_rule_fired,
        "rules_triggered": rule_contributions,
        "input_memberships": input_memberships,
        "aggregated_output_shape": {
            "x": universe.tolist(),
            "y": aggregated_mf.tolist(),
        },
    }


@app.get("/plot/{var_name}")
async def get_plot_data(var_name: str, b: FuzzyBuilder = Depends(get_builder)):
    var = b.get_variable(var_name)
    x = torch.linspace(var.range_min, var.range_max, 200)
    with torch.no_grad():
        terms = {t_name: mf(x).tolist() for t_name, mf in var.terms.items()}
    return {"x": x.tolist(), "terms": terms}


# Mount UI
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui"), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)))
