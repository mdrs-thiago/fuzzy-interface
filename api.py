import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Any
import base64
import io
import pandas as pd
from fuzzy_torch import FuzzyModule, FuzzyVariable, FuzzyRule, TriangularMF, TrapezoidalMF, GaussianMF, BellMF, SigmoidMF
from fuzzy_torch.defuzz import Centroid
from eautomfis.wang_mendel import WangMendelExtractor

app = FastAPI(title="Fuzzy Inference System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State (Builder Pattern) ---
class FuzzyBuilder:
    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.roles: Dict[str, str] = {} # var_name -> 'input' | 'output'
        self.rules_data: List[Dict] = [] # Store raw rule config for reconstruction
        self.system: Optional[FuzzyModule] = None
        self.dirty = False # Flag to rebuild system

    def get_variable(self, name: str):
        if name not in self.variables:
            raise HTTPException(status_code=404, detail=f"Variable '{name}' not found")
        return self.variables[name]

    def delete_variable(self, name: str):
        if name in self.variables:
            del self.variables[name]
        if name in self.roles:
            del self.roles[name]
        # Remove rules referencing this variable
        self.rules_data = [
            r for r in self.rules_data 
            if name not in r['antecedents'] and name not in r['consequent']
        ]
        self.system = None

    def delete_term(self, var_name: str, term_name: str):
        var = self.get_variable(var_name)
        if term_name in var.terms:
            del var.terms[term_name]
        # Remove rules referencing this term
        self.rules_data = [
            r for r in self.rules_data 
            if r['antecedents'].get(var_name) != term_name and r['consequent'].get(var_name) != term_name
        ]
        self.system = None

    def delete_rule(self, index: int):
        if 0 <= index < len(self.rules_data):
            self.rules_data.pop(index)
        self.system = None

builder = FuzzyBuilder()

# --- Pydantic Models ---
class CreateVariable(BaseModel):
    name: str
    min_val: float
    max_val: float
    role: str = "input" # input or output

class CreateMF(BaseModel):
    name: str
    type: str  # tri, trap, gauss, bell, sig
    params: List[float]

class CreateRule(BaseModel):
    antecedents: Dict[str, str] # {VarName: TermName}
    consequent: Dict[str, str]  # {VarName: TermName} (Single consequent for now)

class SimulateRequest(BaseModel):
    inputs: Dict[str, float]

class SystemConfig(BaseModel):
    variables: Dict[str, CreateVariable] # Reuse logic roughly, or define better structure. 
    # Actually, export format might differ from create logic.
    # Let's define a schema for Variable Dump
    pass

class WangMendelRequest(BaseModel):
    csv_data: str # base64 encoded csv
    target_col: str
    n_terms: int = 5

# --- Endpoints ---

@app.post("/wang-mendel")
def run_wang_mendel(req: WangMendelRequest):
    try:
        csv_str = base64.b64decode(req.csv_data).decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_str))
        
        if req.target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{req.target_col}' not found in data.")
            
        extractor = WangMendelExtractor(n_terms=req.n_terms)
        extractor.fit(df, req.target_col)
        config = extractor.export_config(req.target_col)
        return config
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
def get_config():
    """Returns current system configuration."""
    vars_out = {}
    for name, var in builder.variables.items():
        terms = {}
        for t_name, mf in var.terms.items():
            # Extract params based on type
            params = []
            mf_type = type(mf).__name__
            if isinstance(mf, TriangularMF):
                params = [mf.a.item(), mf.b.item(), mf.c.item()]
            elif isinstance(mf, TrapezoidalMF):
                params = [mf.a.item(), mf.b.item(), mf.c.item(), mf.d.item()]
            elif isinstance(mf, GaussianMF):
                params = [mf.mu.item(), mf.sigma.item()]
            
            terms[t_name] = {"type": mf_type, "params": params}
            
        vars_out[name] = {
            "min": var.range_min,
            "max": var.range_max,
            "role": builder.roles.get(name, "input"),
            "terms": terms
        }
        
    return {
        "variables": vars_out,
        "rules": builder.rules_data
    }

@app.get("/export")
def export_system():
    """Export the full system configuration."""
    # We can reuse the get_config logic but structure it for re-import
    # Or just return the raw builder state that creates it.
    
    # Let's try to return a structure that mimics what we need to rebuild.
    # For now, let's just dump the internal state in a clean JSON way.
    
    # 1. Variables & Terms
    vars_export = {}
    for name, var in builder.variables.items():
        terms = []
        for t_name, mf in var.terms.items():
            params = []
            mf_type = type(mf).__name__
            if isinstance(mf, TriangularMF): params = [mf.a.item(), mf.b.item(), mf.c.item()]
            elif isinstance(mf, TrapezoidalMF): params = [mf.a.item(), mf.b.item(), mf.c.item(), mf.d.item()]
            elif isinstance(mf, GaussianMF): params = [mf.mu.item(), mf.sigma.item()]
            # ... others
            terms.append({"name": t_name, "type": mf_type, "params": params})
            
        vars_export[name] = {
            "min": var.range_min,
            "max": var.range_max,
            "role": builder.roles.get(name, "input"),
            "terms": terms
        }
        
    return {
        "variables": vars_export,
        "rules": builder.rules_data
    }

@app.post("/import")
def import_system(config: Dict[str, Any]):
    """Restores system from config."""
    try:
        # Reset first
        builder.variables = {}
        builder.roles = {}
        builder.rules_data = []
        builder.system = None
        
        # 1. Restore Variables
        for name, data in config.get("variables", {}).items():
            var = FuzzyVariable(name, data["min"], data["max"])
            builder.variables[name] = var
            builder.roles[name] = data.get("role", "input")
            
            # Restore Terms
            for term in data.get("terms", []):
                mf = None
                p = term["params"]
                t_type = term["type"]
                # Mapping type names from export back to classes
                if "TriangularMF" in t_type or t_type == "tri": mf = TriangularMF(p[0], p[1], p[2])
                elif "TrapezoidalMF" in t_type or t_type == "trap": mf = TrapezoidalMF(p[0], p[1], p[2], p[3])
                elif "GaussianMF" in t_type or t_type == "gauss": mf = GaussianMF(p[0], p[1])
                # ... others
                
                if mf: var.add_term(term["name"], mf)
                
        # 2. Restore Rules
        builder.rules_data = config.get("rules", [])
        
        return {"message": "System imported successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

@app.post("/reset")
def reset_system():
    builder.variables = {}
    builder.roles = {}
    builder.rules_data = []
    builder.system = None
    return {"message": "System reset"}

@app.post("/preset/{scenario_id}")
def load_preset(scenario_id: str):
    """Loads a FIS preset for a specific laboratory scenario."""
    builder.variables = {}
    builder.roles = {}
    builder.rules_data = []
    builder.system = None

    if scenario_id == "dc_motor":
        error = FuzzyVariable("error", -100, 100)
        builder.variables["error"] = error
        builder.roles["error"] = "input"

        error_dot = FuzzyVariable("error_dot", -50, 50)
        builder.variables["error_dot"] = error_dot
        builder.roles["error_dot"] = "input"

        voltage = FuzzyVariable("voltage", -12, 12)
        builder.roles["voltage"] = "output"
        builder.variables["voltage"] = voltage

    elif scenario_id == "tank_broken":
        level = FuzzyVariable("level", -1, 1)
        level.add_term("low", TriangularMF(-1, -1, 0))
        level.add_term("okay", TriangularMF(-0.5, 0, 0.5)) # Deliberately broken (too wide)
        level.add_term("high", TriangularMF(0, 1, 1))
        builder.variables["level"] = level
        builder.roles["level"] = "input"

        rate = FuzzyVariable("rate", -0.1, 0.1)
        rate.add_term("negative", TriangularMF(-0.1, -0.1, 0))
        rate.add_term("none", TriangularMF(-0.01, 0, 0.01))
        rate.add_term("positive", TriangularMF(0, 0.1, 0.1))
        builder.variables["rate"] = rate
        builder.roles["rate"] = "input"

        valve = FuzzyVariable("valve", -1, 1)
        valve.add_term("close_fast", TriangularMF(-1, -1, -0.3))
        valve.add_term("close_slow", TriangularMF(-0.6, -0.3, 0))
        valve.add_term("no_change", TriangularMF(-0.1, 0, 0.1))
        valve.add_term("open_slow", TriangularMF(0, 0.2, 0.5))
        valve.add_term("open_fast", TriangularMF(0.3, 1, 1))
        builder.variables["valve"] = valve
        builder.roles["valve"] = "output"

        # Broken rules: swapped consequents and missing rule
        builder.rules_data = [
            {"antecedents": {"level": "okay"}, "consequent": {"valve": "no_change"}},
            {"antecedents": {"level": "low"}, "consequent": {"valve": "close_fast"}},
            {"antecedents": {"level": "high"}, "consequent": {"valve": "open_fast"}},
            {"antecedents": {"level": "okay", "rate": "positive"}, "consequent": {"valve": "close_slow"}},
        ]

    elif scenario_id == "tank":
        # Input 1: level error (setpoint - current), normalized [-1, 1]
        level = FuzzyVariable("level", -1, 1)
        level.add_term("low", TriangularMF(-1, -1, 0))
        level.add_term("okay", TriangularMF(-0.1, 0, 0.1))
        level.add_term("high", TriangularMF(0, 1, 1))
        builder.variables["level"] = level
        builder.roles["level"] = "input"

        # Input 2: rate of change of water level
        rate = FuzzyVariable("rate", -0.1, 0.1)
        rate.add_term("negative", TriangularMF(-0.1, -0.1, 0))
        rate.add_term("none", TriangularMF(-0.01, 0, 0.01))
        rate.add_term("positive", TriangularMF(0, 0.1, 0.1))
        builder.variables["rate"] = rate
        builder.roles["rate"] = "input"

        # Output: valve rate of change [-1, 1], 5 MFs (asymmetric)
        valve = FuzzyVariable("valve", -1, 1)
        valve.add_term("close_fast", TriangularMF(-1, -1, -0.3))
        valve.add_term("close_slow", TriangularMF(-0.6, -0.3, 0))
        valve.add_term("no_change", TriangularMF(-0.1, 0, 0.1))
        valve.add_term("open_slow", TriangularMF(0, 0.2, 0.5))
        valve.add_term("open_fast", TriangularMF(0.3, 1, 1))
        builder.variables["valve"] = valve
        builder.roles["valve"] = "output"

        # 5 Rules (MathWorks sltank)
        builder.rules_data = [
            {"antecedents": {"level": "okay"}, "consequent": {"valve": "no_change"}},
            {"antecedents": {"level": "low"}, "consequent": {"valve": "open_fast"}},
            {"antecedents": {"level": "high"}, "consequent": {"valve": "close_fast"}},
            {"antecedents": {"level": "okay", "rate": "positive"}, "consequent": {"valve": "close_slow"}},
            {"antecedents": {"level": "okay", "rate": "negative"}, "consequent": {"valve": "open_slow"}}
        ]

    elif scenario_id == "crane":
        # Inputs
        pos = FuzzyVariable("PosErr", -10, 10)
        pos.add_term("Neg", TriangularMF(-10, -10, 0))
        pos.add_term("Zero", TriangularMF(-1, 0, 1))
        pos.add_term("Pos", TriangularMF(0, 10, 10))
        builder.variables["PosErr"] = pos
        builder.roles["PosErr"] = "input"

        vel = FuzzyVariable("Vel", -5, 5)
        vel.add_term("Neg", TriangularMF(-5, -5, 0))
        vel.add_term("Zero", TriangularMF(-0.5, 0, 0.5))
        vel.add_term("Pos", TriangularMF(0, 5, 5))
        builder.variables["Vel"] = vel
        builder.roles["Vel"] = "input"

        angle = FuzzyVariable("Angle", -1, 1)
        angle.add_term("Neg", TriangularMF(-1, -1, 0))
        angle.add_term("Zero", TriangularMF(-0.1, 0, 0.1))
        angle.add_term("Pos", TriangularMF(0, 1, 1))
        builder.variables["Angle"] = angle
        builder.roles["Angle"] = "input"

        w = FuzzyVariable("W", -2, 2)
        w.add_term("Neg", TriangularMF(-2, -2, 0))
        w.add_term("Zero", TriangularMF(-0.2, 0, 0.2))
        w.add_term("Pos", TriangularMF(0, 2, 2))
        builder.variables["W"] = w
        builder.roles["W"] = "input"

        # Output
        f = FuzzyVariable("Force", -30, 30)
        f.add_term("Left", TriangularMF(-30, -30, 0))
        f.add_term("Zero", TriangularMF(-2, 0, 2))
        f.add_term("Right", TriangularMF(0, 30, 30))
        builder.variables["Force"] = f
        builder.roles["Force"] = "output"

        # Tuned Rules
        builder.rules_data = [
            {"antecedents": {"PosErr": "Pos"}, "consequent": {"Force": "Right"}},
            {"antecedents": {"PosErr": "Neg"}, "consequent": {"Force": "Left"}},
            {"antecedents": {"Vel": "Pos"}, "consequent": {"Force": "Left"}},
            {"antecedents": {"Vel": "Neg"}, "consequent": {"Force": "Right"}},
            {"antecedents": {"Angle": "Pos"}, "consequent": {"Force": "Left"}},
            {"antecedents": {"Angle": "Neg"}, "consequent": {"Force": "Right"}},
            {"antecedents": {"W": "Pos"}, "consequent": {"Force": "Left"}},
            {"antecedents": {"W": "Neg"}, "consequent": {"Force": "Right"}}
        ]

    elif scenario_id == "robot_linear":
        # Distance error and its rate of change
        dist = FuzzyVariable("DistErr", -10, 10)
        dist.add_term("Neg", TriangularMF(-10, -10, 0))
        dist.add_term("Zero", TriangularMF(-1, 0, 1))
        dist.add_term("Pos", TriangularMF(0, 10, 10))
        builder.variables["DistErr"] = dist
        builder.roles["DistErr"] = "input"

        rate = FuzzyVariable("Rate", -2, 2)
        rate.add_term("Neg", TriangularMF(-2, -2, 0))
        rate.add_term("Zero", TriangularMF(-0.3, 0, 0.3))
        rate.add_term("Pos", TriangularMF(0, 2, 2))
        builder.variables["Rate"] = rate
        builder.roles["Rate"] = "input"

        power = FuzzyVariable("Power", -1, 1)
        power.add_term("Reverse", TriangularMF(-1, -1, 0))
        power.add_term("Stop", TriangularMF(-0.2, 0, 0.2))
        power.add_term("Forward", TriangularMF(0, 1, 1))
        builder.variables["Power"] = power
        builder.roles["Power"] = "output"

        builder.rules_data = [
            {"antecedents": {"DistErr": "Pos"}, "consequent": {"Power": "Forward"}},
            {"antecedents": {"DistErr": "Neg"}, "consequent": {"Power": "Reverse"}},
            {"antecedents": {"DistErr": "Zero"}, "consequent": {"Power": "Stop"}},
            {"antecedents": {"DistErr": "Zero", "Rate": "Pos"}, "consequent": {"Power": "Reverse"}},
            {"antecedents": {"DistErr": "Zero", "Rate": "Neg"}, "consequent": {"Power": "Forward"}}
        ]

    elif scenario_id == "robot_angular":
        # Angular velocity error and its rate
        ang_err = FuzzyVariable("AngErr", -5, 5)
        ang_err.add_term("Neg", TriangularMF(-5, -5, 0))
        ang_err.add_term("Zero", TriangularMF(-0.5, 0, 0.5))
        ang_err.add_term("Pos", TriangularMF(0, 5, 5))
        builder.variables["AngErr"] = ang_err
        builder.roles["AngErr"] = "input"

        ang_rate = FuzzyVariable("AngRate", -2, 2)
        ang_rate.add_term("Neg", TriangularMF(-2, -2, 0))
        ang_rate.add_term("Zero", TriangularMF(-0.3, 0, 0.3))
        ang_rate.add_term("Pos", TriangularMF(0, 2, 2))
        builder.variables["AngRate"] = ang_rate
        builder.roles["AngRate"] = "input"

        steer = FuzzyVariable("Steering", -1, 1)
        steer.add_term("Left", TriangularMF(-1, -1, 0))
        steer.add_term("None", TriangularMF(-0.2, 0, 0.2))
        steer.add_term("Right", TriangularMF(0, 1, 1))
        builder.variables["Steering"] = steer
        builder.roles["Steering"] = "output"

        builder.rules_data = [
            {"antecedents": {"AngErr": "Pos"}, "consequent": {"Steering": "Right"}},
            {"antecedents": {"AngErr": "Neg"}, "consequent": {"Steering": "Left"}},
            {"antecedents": {"AngErr": "Zero"}, "consequent": {"Steering": "None"}},
            {"antecedents": {"AngErr": "Zero", "AngRate": "Pos"}, "consequent": {"Steering": "Left"}},
            {"antecedents": {"AngErr": "Zero", "AngRate": "Neg"}, "consequent": {"Steering": "Right"}}
        ]

    elif scenario_id == "robot_nav":
        # 4 inputs: distance error, dDist, heading error, dHeading
        dist = FuzzyVariable("DistErr", 0, 10)
        dist.add_term("Close", TriangularMF(0, 0, 2))
        dist.add_term("Medium", TriangularMF(1, 5, 9))
        dist.add_term("Far", TriangularMF(7, 10, 10))
        builder.variables["DistErr"] = dist
        builder.roles["DistErr"] = "input"

        ddist = FuzzyVariable("dDist", -2, 2)
        ddist.add_term("Neg", TriangularMF(-2, -2, 0))
        ddist.add_term("Zero", TriangularMF(-0.3, 0, 0.3))
        ddist.add_term("Pos", TriangularMF(0, 2, 2))
        builder.variables["dDist"] = ddist
        builder.roles["dDist"] = "input"

        head = FuzzyVariable("HeadErr", -3.15, 3.15)
        head.add_term("Neg", TriangularMF(-3.15, -3.15, 0))
        head.add_term("Zero", TriangularMF(-0.3, 0, 0.3))
        head.add_term("Pos", TriangularMF(0, 3.15, 3.15))
        builder.variables["HeadErr"] = head
        builder.roles["HeadErr"] = "input"

        dhead = FuzzyVariable("dHead", -1, 1)
        dhead.add_term("Neg", TriangularMF(-1, -1, 0))
        dhead.add_term("Zero", TriangularMF(-0.15, 0, 0.15))
        dhead.add_term("Pos", TriangularMF(0, 1, 1))
        builder.variables["dHead"] = dhead
        builder.roles["dHead"] = "input"

        # Output: linear velocity [0, 1]
        vel = FuzzyVariable("Velocity", 0, 1)
        vel.add_term("Stop", TriangularMF(0, 0, 0.2))
        vel.add_term("Slow", TriangularMF(0.1, 0.3, 0.5))
        vel.add_term("Fast", TriangularMF(0.4, 1, 1))
        builder.variables["Velocity"] = vel
        builder.roles["Velocity"] = "output"

        builder.rules_data = [
            {"antecedents": {"DistErr": "Far", "HeadErr": "Zero"}, "consequent": {"Velocity": "Fast"}},
            {"antecedents": {"DistErr": "Medium"}, "consequent": {"Velocity": "Slow"}},
            {"antecedents": {"DistErr": "Close"}, "consequent": {"Velocity": "Stop"}},
            {"antecedents": {"HeadErr": "Pos"}, "consequent": {"Velocity": "Slow"}},
            {"antecedents": {"HeadErr": "Neg"}, "consequent": {"Velocity": "Slow"}}
        ]

    else:
        raise HTTPException(status_code=404, detail="Scenario not found")

    return {"message": f"Preset for {scenario_id} loaded"}

@app.post("/variable")
def create_variable(data: CreateVariable):
    builder.variables[data.name] = FuzzyVariable(data.name, data.min_val, data.max_val)
    builder.roles[data.name] = data.role
    return {"message": f"Variable {data.name} created"}

@app.delete("/variable/{name}")
def delete_variable(name: str):
    builder.delete_variable(name)
    return {"message": f"Variable {name} deleted"}

@app.delete("/variable/{name}/term/{term}")
def delete_term(name: str, term: str):
    builder.delete_term(name, term)
    return {"message": f"Term {term} deleted from {name}"}

@app.delete("/rule/{index}")
def delete_rule(index: int):
    builder.delete_rule(index)
    return {"message": f"Rule {index} deleted"}

@app.post("/variable/{name}/term")
def add_term(name: str, data: CreateMF):
    var = builder.get_variable(name)
    
    mf = None
    p = data.params
    if data.type == "tri":
        mf = TriangularMF(p[0], p[1], p[2])
    elif data.type == "trap":
        mf = TrapezoidalMF(p[0], p[1], p[2], p[3])
    elif data.type == "gauss":
        mf = GaussianMF(p[0], p[1])
    elif data.type == "bell":
        mf = BellMF(p[0], p[1], p[2])
    elif data.type == "sig":
        mf = SigmoidMF(p[0], p[1])
    else:
        raise HTTPException(status_code=400, detail="Unknown MF type")
        
    var.add_term(data.name, mf)
    return {"message": f"Term {data.name} added to {name}"}

@app.post("/rule")
def add_rule(data: CreateRule):
    # Just store for now, validate existence
    for v, t in data.antecedents.items():
        if v not in builder.variables or t not in builder.variables[v].terms:
             raise HTTPException(status_code=400, detail=f"Invalid antecedent {v}:{t}")
    
    cons_var = list(data.consequent.keys())[0]
    cons_term = list(data.consequent.values())[0]
    if cons_var not in builder.variables or cons_term not in builder.variables[cons_var].terms:
        raise HTTPException(status_code=400, detail=f"Invalid consequent {cons_var}:{cons_term}")

    builder.rules_data.append(data.dict())
    builder.system = None # Invalidate built system
    return {"message": "Rule added"}

@app.post("/build")
def build_system():
    """Builds the system using defined input/output roles."""
    
    # Identify output logic
    outputs = [n for n, r in builder.roles.items() if r == 'output']
    if not outputs:
        # Fallback if no output defined explicitly? Or error.
        # Let's try to auto-detect from rules? No, enforce role.
        raise HTTPException(status_code=400, detail="No output variable defined. Please create a variable with role='output'.")
    
    output_var_name = outputs[0] # Single output supported for now
    
    inputs = [v for k, v in builder.variables.items() if builder.roles.get(k) == 'input']
    output = builder.variables[output_var_name]
    
    rules = []
    for r_data in builder.rules_data:
        antecedents = []
        for v_name, t_name in r_data['antecedents'].items():
            antecedents.append((builder.variables[v_name], t_name))
        
        # Consequent
        c_name = list(r_data['consequent'].keys())[0]
        c_term = list(r_data['consequent'].values())[0]
        
        rules.append(FuzzyRule(antecedent=antecedents, consequent=(builder.variables[c_name], c_term)))
        
    builder.system = FuzzyModule(inputs=inputs, output=output, rules=rules, defuzz_method=Centroid(output.range_min, output.range_max))
    return {"message": "System built successfully"}

@app.post("/simulate")
def simulate(data: SimulateRequest):
    if not builder.system:
        raise HTTPException(status_code=400, detail="System not built yet")
        
    # Convert inputs to tensor dict
    inputs_torch = {k: torch.tensor([v]) for k, v in data.inputs.items()}
    
    try:
        # Get Activations
        activations = builder.system.get_rule_activations(inputs_torch)[0] # (num_rules,)
        
        # Get Output
        output = builder.system(inputs_torch)
        
        input_memberships = {}
        for var_name, val in data.inputs.items():
            if var_name in builder.variables:
                var = builder.variables[var_name]
                input_memberships[var_name] = {}
                for t_name, mf in var.terms.items():
                    input_memberships[var_name][t_name] = mf(torch.tensor([val])).item()

        universe = builder.system.defuzz.universe
        aggregated_mf = torch.zeros(builder.system.defuzz.resolution, device=universe.device)
        
        rule_contributions = []
        for i, rule in enumerate(builder.system.rules):
            strength = activations[i].item()
            if strength > 1e-4:
                full_cons_term = builder.system.output_variable.terms[rule.consequent[1]]
                cons_mf_values = full_cons_term(universe)
                rule_output = torch.min(torch.tensor([strength]), cons_mf_values)
                
                rd = builder.rules_data[i]
                parts = [f"{v} IS {t}" for v, t in rd.get("antecedents", {}).items()]
                desc = f"IF {' AND '.join(parts)} THEN {list(rd.get('consequent', {}).keys())[0]} IS {list(rd.get('consequent', {}).values())[0]}"
                
                rule_contributions.append({
                    "index": i,
                    "description": desc,
                    "config": rd,
                    "strength": strength,
                    "consequent_term": rule.consequent[1],
                    "clipped_shape": rule_output.tolist()
                })
                
                if i == 0:
                    aggregated_mf = rule_output
                else:
                    aggregated_mf = builder.system.aggregation(aggregated_mf, rule_output)
                
        return {
            "output": output.item(),
            "rules_triggered": rule_contributions,
            "input_memberships": input_memberships,
            "aggregated_output_shape": {
                "x": universe.tolist(),
                "y": aggregated_mf.tolist()
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plot/{var_name}")
def get_plot_data(var_name: str):
    var = builder.get_variable(var_name)
    x = torch.linspace(var.range_min, var.range_max, 200)
    data = {"x": x.tolist(), "terms": {}}
    
    for t_name, mf in var.terms.items():
        y = mf(x)
        data["terms"][t_name] = y.tolist()
        
    return data

# Mount UI
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
